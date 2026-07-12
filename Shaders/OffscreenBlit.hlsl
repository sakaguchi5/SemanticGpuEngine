Texture2D<float4> SourceTexture : register(t0);

struct VertexInput
{
    float3 position : POSITION;
    float4 color : COLOR;
};

struct PixelInput
{
    float4 position : SV_Position;
};

PixelInput VSMain(VertexInput input)
{
    PixelInput output;
    output.position = float4(input.position, 1.0f);
    return output;
}

float4 PSMain(PixelInput input) : SV_Target
{
    return SourceTexture.Load(int3(int2(input.position.xy), 0));
}
