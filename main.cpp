#include <OpenCL/cl.hpp>
#include "fstream"
#include "iostream"
#include "numeric"

cl::Program createProgram(const std::string& file) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    assert(platforms.size() > 0);

    auto platform = platforms.front();

    std::vector<cl::Device> devices;
    platforms.front().getDevices(CL_DEVICE_TYPE_GPU, &devices);

    assert(devices.size() > 0);

    auto device = devices.front();
    auto vendor = device.getInfo<CL_DEVICE_VENDOR>();
    auto version = device.getInfo<CL_DEVICE_VERSION>();

    std::ifstream kernelFile(file);
    std::string src(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));

    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

    cl::Context context(device);
    cl::Program program(context, sources);

    auto err = program.build("-cl-std=CL1.2");

    return program;
}

void HelloWorld() {
    auto program = createProgram("HelloWorld.cl");
    auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
    auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
    auto& device = devices.front();

    char buf[16];
    cl::Buffer memBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(buf));
    cl_int err = 0;
    cl::Kernel kernel(program, "HelloWorld", &err);
    kernel.setArg(0, memBuf);

    cl::CommandQueue queue(context, device);
    queue.enqueueTask(kernel);
    queue.enqueueReadBuffer(memBuf, CL_TRUE, 0, sizeof(buf), buf);

    std::cout << buf;
}

void ProcessArray() {
    auto program = createProgram("ProcessArray.cl");
    auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
    auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
    auto& device = devices.front();

    std::vector<int> vec(1024);
    std::iota(vec.begin(), vec.end(), 1);

    std::vector<int> vecOut(1024);

    cl_int err = 0;

    cl::Buffer inBuf(context,
                     CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                     sizeof(int) * vec.size(),
                     vec.data(),
                     &err);
    cl::Buffer outBuf(context,
                      CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                      sizeof(int) * vecOut.size(),
                      nullptr,
                      &err);

    cl::Kernel kernel(program, "ProcessArray", &err);
    err = kernel.setArg(0, inBuf);
    err = kernel.setArg(1, outBuf);

    cl::CommandQueue queue(context, device);

    err = queue.enqueueFillBuffer(inBuf, 3, sizeof(int) * 5, sizeof(int) * (vec.size() - 1000));
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vec.size()));
    err = queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, sizeof(int) * vecOut.size(), vecOut.data());

    cl::finish();

    std::cout << vecOut.front() << std::endl;
}


int main() {
//    HelloWorld();
    ProcessArray();

    return 0;
}
