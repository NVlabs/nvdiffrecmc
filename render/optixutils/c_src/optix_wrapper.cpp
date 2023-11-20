// Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifdef _MSC_VER 
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <algorithm>
#include <fstream>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <nvrtc.h>
#include <cuda_runtime.h>

#include "common.h"
#include "optix_wrapper.h"

// NVRTC compiler options
#define CUDA_NVRTC_OPTIONS  \
  "-std=c++11", \
  "-arch", \
  "compute_70", \
  "-use_fast_math", \
  "-lineinfo", \
  "-default-device", \
  "-rdc", \
  "true", \
  "-D__x86_64", \
  "-D__OPTIX__"

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}

static bool readSourceFile( std::string& str, const std::string& filename )
{
    // Try to open file
    std::ifstream file( filename.c_str(), std::ios::binary );
    if( file.good() )
    {
        // Found usable source file
        std::vector<unsigned char> buffer = std::vector<unsigned char>( std::istreambuf_iterator<char>( file ), {} );
        str.assign(buffer.begin(), buffer.end());
        return true;
    }
    return false;
}

static void getCuStringFromFile( std::string& cu, const char* filename )
{
    // Try to get source code from file
    if( readSourceFile( cu, filename ) )
        return;

    // Wasn't able to find or open the requested file
    throw std::runtime_error( "Couldn't open source file " + std::string( filename ) );
}

static void getPtxFromCuString( std::string& ptx, const char* include_dir, const char* optix_include_dir, const char* cuda_include_dir, const char* cu_source, 
    const char* name, const char** log_string )
{
    // Create program
    nvrtcProgram prog = 0;
    NVRTC_CHECK_ERROR( nvrtcCreateProgram( &prog, cu_source, name, 0, NULL, NULL ) );

    // Gather NVRTC options
    std::vector<const char*> options;

    std::string sample_dir;
    sample_dir = std::string( "-I" ) + include_dir;
    options.push_back( sample_dir.c_str() );

    // Collect include dirs
    std::vector<std::string> include_dirs;
    
    include_dirs.push_back( std::string( "-I" ) + optix_include_dir );
    include_dirs.push_back( std::string( "-I" ) + cuda_include_dir );

    for( const std::string& dir : include_dirs)
        options.push_back( dir.c_str() );

    // Collect NVRTC options
    const char*  compiler_options[] = {CUDA_NVRTC_OPTIONS};
    std::copy( std::begin( compiler_options ), std::end( compiler_options ), std::back_inserter( options ) );

    // JIT compile CU to PTX
    const nvrtcResult compileRes = nvrtcCompileProgram( prog, (int)options.size(), options.data() );

    // Retrieve log output
    std::string g_nvrtcLog;
    size_t log_size = 0;
    NVRTC_CHECK_ERROR( nvrtcGetProgramLogSize( prog, &log_size ) );
    g_nvrtcLog.resize( log_size );
    if( log_size > 1 )
    {
        NVRTC_CHECK_ERROR( nvrtcGetProgramLog( prog, &g_nvrtcLog[0] ) );
        if( log_string )
            *log_string = g_nvrtcLog.c_str();
    }
    if( compileRes != NVRTC_SUCCESS )
        throw std::runtime_error( "NVRTC Compilation failed.\n" + g_nvrtcLog );

    // Retrieve PTX code
    size_t ptx_size = 0;
    NVRTC_CHECK_ERROR( nvrtcGetPTXSize( prog, &ptx_size ) );
    ptx.resize( ptx_size );
    NVRTC_CHECK_ERROR( nvrtcGetPTX( prog, &ptx[0] ) );

    // Cleanup
    NVRTC_CHECK_ERROR( nvrtcDestroyProgram( &prog ) );
}

const char* getInputData( const char* filename, const char* include_dir, const char* optix_include_dir, 
                          const char* cuda_include_dir, const char* name, size_t& dataSize, const char** log)
{
    if( log )
        *log = NULL;

    std::string * ptx, cu;
    ptx = new std::string();

    getCuStringFromFile( cu, filename );
    getPtxFromCuString( *ptx, include_dir, optix_include_dir, cuda_include_dir, cu.c_str(), name, log );

    dataSize = ptx->size();
    return ptx->c_str();
}


struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

void createPipeline(const OptixDeviceContext context, const std::string& path, const std::string& cuda_path, 
                const std::string& kernel_name, OptixModule* module, OptixPipeline* pipeline, OptixShaderBindingTable& sbt)
{
    char log[2048];
    OptixPipelineCompileOptions pipeline_compile_options = {};
    {
        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;

        pipeline_compile_options.usesMotionBlur        = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options.numPayloadValues      = 1;
        pipeline_compile_options.numAttributeValues    = 2;
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

        size_t      inputSize  = 0;
        std::string shaderFile = path + "/c_src/" + kernel_name + "/kernel.cu";
        std::string includeDir = path + "/c_src/" + kernel_name;
        std::string optix_include_dir = path + "/include";
        std::string cuda_include_dir = cuda_path + "/include";

        const char* input = getInputData(shaderFile.c_str(), includeDir.c_str(), optix_include_dir.c_str(),
                                         cuda_include_dir.c_str(), "kernel", inputSize, (const char**)&log);
        size_t sizeof_log = sizeof( log );

        OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                    context,
                    &module_compile_options,
                    &pipeline_compile_options,
                    input,
                    inputSize,
                    log,
                    &sizeof_log,
                    module) );
    }

    //
    // Create program groups
    //
    OptixProgramGroup raygen_prog_group   = nullptr;
    OptixProgramGroup miss_prog_group     = nullptr;
    {
        OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros

        OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = *module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
        size_t sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    context,
                    &raygen_prog_group_desc,
                    1,   // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &raygen_prog_group
                    ) );

        OptixProgramGroupDesc miss_prog_group_desc  = {};
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = *module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
        sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    context,
                    &miss_prog_group_desc,
                    1,   // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &miss_prog_group
                    ) );
    }

    //
    // Link pipeline
    //
    {
        const uint32_t    max_trace_depth  = 1;
        OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group };

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth          = max_trace_depth;
        pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
        size_t sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixPipelineCreate(
                    context,
                    &pipeline_compile_options,
                    &pipeline_link_options,
                    program_groups,
                    sizeof( program_groups ) / sizeof( program_groups[0] ),
                    log,
                    &sizeof_log,
                    pipeline
                    ) );

        OptixStackSizes stack_sizes = {};
        for( auto& prog_group : program_groups )
        {
            OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes ) );
        }

        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
                                                 0,  // maxCCDepth
                                                 0,  // maxDCDEpth
                                                 &direct_callable_stack_size_from_traversal,
                                                 &direct_callable_stack_size_from_state, &continuation_stack_size ) );
        OPTIX_CHECK( optixPipelineSetStackSize( *pipeline, direct_callable_stack_size_from_traversal,
                                                direct_callable_stack_size_from_state, continuation_stack_size,
                                                1  // maxTraversableDepth
                                                ) );
    }

    //
    // Set up shader binding table
    //
    {
        CUdeviceptr  raygen_record;
        const size_t raygen_record_size = sizeof( SbtRecord );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
        SbtRecord rg_sbt;
        OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( raygen_record ),
                    &rg_sbt,
                    raygen_record_size,
                    cudaMemcpyHostToDevice
                    ) );

        CUdeviceptr miss_record;
        size_t      miss_record_size = sizeof( SbtRecord );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
        SbtRecord ms_sbt;
        OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( miss_record ),
                    &ms_sbt,
                    miss_record_size,
                    cudaMemcpyHostToDevice
                    ) );

        sbt.raygenRecord                = raygen_record;
        sbt.missRecordBase              = miss_record;
        sbt.missRecordStrideInBytes     = sizeof( SbtRecord );
        sbt.missRecordCount             = 1;
    }    
}

OptiXStateWrapper::OptiXStateWrapper(const std::string& path, const std::string& cuda_path)
{
    pState = new OptiXState();
    memset(pState, 0, sizeof(OptiXState));

    // create OptiX context
    pState->context = nullptr;
    {
        // Initialize the OptiX API, loading all API entry points
        OPTIX_CHECK( optixInit() );

        // Specify context options
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction       = &context_log_cb;
        options.logCallbackLevel          = 0;

        // Associate a CUDA context (and therefore a specific GPU) with this
        // device context
        CUcontext cuCtx = 0;  // zero means take the current context
        OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &pState->context ) );
    }

    // Create pipelines
    pState->moduleEnvSampling = nullptr;
    pState->pipelineEnvSampling = nullptr;
    pState->sbtEnvSampling = {};
    createPipeline(pState->context, path, cuda_path, "envsampling", &pState->moduleEnvSampling, &pState->pipelineEnvSampling, pState->sbtEnvSampling);

    printf("End of OptiXStateWrapper \n");
}

OptiXStateWrapper::~OptiXStateWrapper(void)
{
    OPTIX_CHECK( optixPipelineDestroy( pState->pipelineEnvSampling ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( pState->sbtEnvSampling.raygenRecord       ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( pState->sbtEnvSampling.missRecordBase     ) ) );
    OPTIX_CHECK( optixModuleDestroy( pState->moduleEnvSampling ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( pState->d_gas_output_buffer ) ) );
    OPTIX_CHECK( optixDeviceContextDestroy( pState->context ) ); 
    delete pState;
    printf("OptiXStateWrapper destructor \n");
}
