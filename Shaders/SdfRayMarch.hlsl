cbuffer SdfScene : register(b0)
{
    float4 CameraPosition;
    float4 CameraRight;
    float4 CameraUp;
    float4 CameraForward;
    float4 Projection;
    float4 Box;
    float4 RayMarch;
};

struct VSInput
{
    float3 position : POSITION;
    float4 color : COLOR;
};

struct VSOutput
{
    float4 position : SV_POSITION;
    float2 screen : TEXCOORD0;
};

VSOutput VSMain(VSInput input)
{
    VSOutput output;
    output.position = float4(input.position, 1.0f);
    output.screen = input.position.xy;
    return output;
}

float3 RotateObjectSpace(float3 samplePosition)
{
    const float angle = -Projection.z * 0.35f;
    const float sine = sin(angle);
    const float cosine = cos(angle);
    return float3(
        cosine * samplePosition.x - sine * samplePosition.z,
        samplePosition.y,
        sine * samplePosition.x + cosine * samplePosition.z);
}

float BoxDistance(float3 samplePosition)
{
    const float3 local = RotateObjectSpace(samplePosition);
    const float3 q = abs(local) - Box.xyz;
    return length(max(q, 0.0f))
        + min(max(q.x, max(q.y, q.z)), 0.0f);
}

float3 EstimateNormal(float3 samplePosition)
{
    const float epsilon = max(RayMarch.y * 2.0f, 0.0005f);
    return normalize(float3(
        BoxDistance(samplePosition + float3(epsilon, 0, 0))
            - BoxDistance(samplePosition - float3(epsilon, 0, 0)),
        BoxDistance(samplePosition + float3(0, epsilon, 0))
            - BoxDistance(samplePosition - float3(0, epsilon, 0)),
        BoxDistance(samplePosition + float3(0, 0, epsilon))
            - BoxDistance(samplePosition - float3(0, 0, epsilon))));
}

float4 PSMain(VSOutput input) : SV_TARGET
{
    const float2 screen = float2(input.screen.x, -input.screen.y);
    const float3 direction = normalize(
        CameraForward.xyz
        + CameraRight.xyz * screen.x * Projection.x * Projection.y
        + CameraUp.xyz * screen.y * Projection.x);

    float distanceTravelled = 0.0f;
    const uint maximumSteps = (uint)RayMarch.z;
    for (uint step = 0; step < maximumSteps; ++step)
    {
        const float3 samplePosition = CameraPosition.xyz
            + direction * distanceTravelled;
        const float distanceToSurface = BoxDistance(samplePosition);
        if (distanceToSurface < RayMarch.y)
        {
            const float3 normal = EstimateNormal(samplePosition);
            const float3 lightDirection = normalize(float3(-0.45f, 0.8f, -0.3f));
            const float diffuse = saturate(dot(normal, lightDirection));
            const float rim = pow(1.0f - saturate(dot(-direction, normal)), 3.0f);
            const float3 base = float3(0.16f, 0.55f, 0.92f);
            return float4(base * (0.18f + 0.82f * diffuse)
                + rim * float3(0.15f, 0.35f, 0.65f), 1.0f);
        }

        distanceTravelled += max(distanceToSurface, RayMarch.y * 0.5f);
        if (distanceTravelled > RayMarch.x)
        {
            break;
        }
    }

    const float horizon = saturate(0.5f + 0.5f * direction.y);
    return float4(lerp(
        float3(0.015f, 0.022f, 0.04f),
        float3(0.07f, 0.10f, 0.17f), horizon), 1.0f);
}
