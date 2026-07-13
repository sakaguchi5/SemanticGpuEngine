ByteAddressBuffer PreviousHistory : register(t0);

struct VSInput
{
    float3 position : POSITION;
};

struct VSOutput
{
    float4 position : SV_POSITION;
};

VSOutput VSMain(VSInput input)
{
    VSOutput output;
    output.position = float4(input.position, 1.0f);
    return output;
}

float4 PSMain(VSOutput input) : SV_TARGET
{
    const float value = min((float)PreviousHistory.Load(0), 1.0f);
    return float4(0.05f, 0.2f + value * 0.6f, 0.35f, 1.0f);
}
