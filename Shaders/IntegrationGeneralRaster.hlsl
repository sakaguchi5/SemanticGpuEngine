struct VSInput
{
    float3 position : POSITION;
    float4 color : COLOR;
    uint instanceId : SV_InstanceID;
};

struct VSOutput
{
    float4 position : SV_Position;
    float4 color : COLOR;
};

VSOutput VSMain(VSInput input)
{
    VSOutput output;
    float offset = input.instanceId == 0 ? -0.28f : 0.28f;
    output.position = float4(input.position.x + offset, input.position.yz, 1.0f);
    output.color = input.color;
    return output;
}

float4 PSMain(VSOutput input) : SV_Target0
{
    return input.color;
}
