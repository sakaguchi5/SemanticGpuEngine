ByteAddressBuffer PreviousHistory : register(t0);
RWByteAddressBuffer CurrentHistory : register(u0);

[numthreads(1, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint previous = PreviousHistory.Load(0);
    CurrentHistory.Store(0, previous + 1);
}
