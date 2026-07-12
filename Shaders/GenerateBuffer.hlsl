RWByteAddressBuffer OutputBuffer : register(u0);

[numthreads(1, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    OutputBuffer.Store(dispatchThreadId.x * 4, 0x12345678);
}
