RWByteAddressBuffer CurrentHistory : register(u0);

[numthreads(1, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    CurrentHistory.Store(0, dispatchThreadId.x + 1);
}
