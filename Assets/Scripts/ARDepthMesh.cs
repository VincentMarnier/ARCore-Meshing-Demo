using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.Serialization;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

[RequireComponent(typeof(MeshCollider), typeof(MeshFilter))]
public class ARDepthMesh : MonoBehaviour
{
    /// <summary>
    /// Depth processing script.
    /// </summary>
    [SerializeField]
    private ComputeShader _depthProcessingCS;

    /// <summary>
    /// Whether to enable the renderer.
    /// </summary>
    [SerializeField]
    private bool _render = false;

    /// <summary>
    /// Makes sure physics objects don't fall through.
    /// </summary>
    [SerializeField]
    private bool _extendMeshEdges = true;

    [SerializeField]
    private AROcclusionManager _occlusionManager;

    [SerializeField]
    private ARCameraManager _cameraManager;

    // Number of threads used by the compute shader.
    private const int _numThreadsX = 8;
    private const int _numThreadsY = 8;
    private const int _kDepthPixelSkippingX = 2;
    private const int _kDepthPixelSkippingY = 2;
    private const int _normalSamplingOffset = 1;
    private const float _edgeExtensionOffset = 0.5f;
    private const float _edgeExtensionDepthOffset = -0.5f;

    // Holds the vertex and index data of the depth template mesh.
    private Mesh _mesh;
    private bool _initialized = false;
    private MeshCollider _meshCollider;
    private int _vertexFromDepthHandle;
    private int _normalFromVertexHandle;
    private int _numElements;
    private ComputeBuffer _vertexBuffer;
    private ComputeBuffer _normalBuffer;
    private Vector3[] _vertices;
    private Vector3[] _normals;
    private int _depthPixelSkippingX = _kDepthPixelSkippingX;
    private int _depthPixelSkippingY = _kDepthPixelSkippingY;
    private int _meshWidth;
    private int _meshHeight;
    private Texture2D _depthTexture;

    private static int[] GenerateTriangles(int width, int height)
    {
        int[] indices = new int[(height - 1) * (width - 1) * 6];
        int idx = 0;
        for (int y = 0; y < (height - 1); y++)
        {
            for (int x = 0; x < (width - 1); x++)
            {
                //// Unity has a clockwise triangle winding order.
                //// Upper quad triangle
                //// Top left
                int idx0 = (y * width) + x;
                //// Top right
                int idx1 = idx0 + 1;
                //// Bottom left
                int idx2 = idx0 + width;

                //// Lower quad triangle
                //// Top right
                int idx3 = idx1;
                //// Bottom right
                int idx4 = idx2 + 1;
                //// Bottom left
                int idx5 = idx2;

                indices[idx++] = idx0;
                indices[idx++] = idx1;
                indices[idx++] = idx2;
                indices[idx++] = idx3;
                indices[idx++] = idx4;
                indices[idx++] = idx5;
            }
        }

        return indices;
    }

    private void OnDestroy()
    {
        _vertexBuffer.Dispose();
        _normalBuffer.Dispose();
    }

    private void Start()
    {
        if (ARSession.state == ARSessionState.None)
        {
            ARSession.CheckAvailability();
        }
        _meshCollider = GetComponent<MeshCollider>();
        GetComponent<MeshRenderer>().enabled = _render;

        Debug.Assert(_occlusionManager);
        Debug.Assert(_cameraManager);
        _occlusionManager.frameReceived += _OcclusionManager_OnFrameReceived;
    }

    private void _OcclusionManager_OnFrameReceived(AROcclusionFrameEventArgs ev)
    {
        if (_cameraManager.TryGetIntrinsics(out XRCameraIntrinsics cameraIntrinsics))
        {
            if (_occlusionManager.TryAcquireEnvironmentDepthCpuImage(out XRCpuImage depthImage))
            {
                using (depthImage)
                {
                    if (_initialized)
                    {
                        Debug.Log("UPDATINNNNNNG");
                        UpdateRawImage(ref _depthTexture, depthImage);
                        UpdateCollider();
                        UpdateMesh();
                    }
                    else if (ARSession.state == ARSessionState.SessionTracking)
                    {
                        if (_occlusionManager.TryAcquireEnvironmentDepthCpuImage(out XRCpuImage image))
                        {
                            using (image)
                            {
                                _meshWidth = image.width / _depthPixelSkippingX;
                                _meshHeight = image.height / _depthPixelSkippingY;
                            }
                            _numElements = _meshWidth * _meshHeight;
                            
                            Vector2 intrinsicsScale;
                            intrinsicsScale.x = image.width / (float)cameraIntrinsics.resolution.x;
                            intrinsicsScale.y = image.height / (float)cameraIntrinsics.resolution.y;

                            var focalLength = Vector2.Scale(cameraIntrinsics.focalLength, intrinsicsScale);
                            var principalPoint = Vector2.Scale(cameraIntrinsics.principalPoint, intrinsicsScale);
                            var resolution = new Vector2Int(image.width, image.height);
                            XRCameraIntrinsics depthCameraIntrinsics = new XRCameraIntrinsics(focalLength, principalPoint, resolution);
                            UpdateRawImage(ref _depthTexture, depthImage);
                            InitializeComputeShader(ref depthCameraIntrinsics);
                            InitializeMesh();
                            _initialized = true;
                        }
                    }
                }
            }
        }
    }

    private void InitializeMesh()
    {
        // Creates template vertices.
        _vertices = new Vector3[_numElements];
        _normals = new Vector3[_numElements];

        // Creates template vertices for the mesh object.
        for (int y = 0; y < _meshHeight; y++)
        {
            for (int x = 0; x < _meshWidth; x++)
            {
                int index = (y * _meshWidth) + x;
                Vector3 v = new Vector3(x * 0.01f, -y * 0.01f, 0);
                _vertices[index] = v;
                _normals[index] = Vector3.back;
            }
        }

        // Creates template triangle list.
        int[] triangles = GenerateTriangles(_meshWidth, _meshHeight);

        // Creates the mesh object and set all template data.
        _mesh = new Mesh();
        _mesh.MarkDynamic();
        _mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        _mesh.vertices = _vertices;
        _mesh.normals = _normals;
        _mesh.triangles = triangles;
        _mesh.bounds = new Bounds(Vector3.zero, new Vector3(20, 20, 20));
        _mesh.UploadMeshData(false);

        if (_render)
        {
            GetComponent<MeshFilter>().sharedMesh = _mesh;
        }
    }

    private void InitializeComputeShader(ref XRCameraIntrinsics cameraIntrinsics)
    {
        _vertexFromDepthHandle = _depthProcessingCS.FindKernel("VertexFromDepth");
        _normalFromVertexHandle = _depthProcessingCS.FindKernel("NormalFromVertex");

        _vertexBuffer = new ComputeBuffer(_numElements, sizeof(float) * 3);
        _normalBuffer = new ComputeBuffer(_numElements, sizeof(float) * 3);

        // Sets general compute shader variables.
        _depthProcessingCS.SetInt("DepthWidth", cameraIntrinsics.resolution.x);
        _depthProcessingCS.SetInt("DepthHeight", cameraIntrinsics.resolution.y);
        _depthProcessingCS.SetFloat("PrincipalX", cameraIntrinsics.principalPoint.x);
        _depthProcessingCS.SetFloat("PrincipalY", cameraIntrinsics.principalPoint.y);
        _depthProcessingCS.SetFloat("FocalLengthX", cameraIntrinsics.focalLength.x);
        _depthProcessingCS.SetFloat("FocalLengthY", cameraIntrinsics.focalLength.y);
        _depthProcessingCS.SetInt("NormalSamplingOffset", _normalSamplingOffset);
        _depthProcessingCS.SetInt("DepthPixelSkippingX", _depthPixelSkippingX);
        _depthProcessingCS.SetInt("DepthPixelSkippingY", _depthPixelSkippingY);
        _depthProcessingCS.SetInt("MeshWidth", _meshWidth);
        _depthProcessingCS.SetInt("MeshHeight", _meshHeight);
        _depthProcessingCS.SetBool("ExtendEdges", _extendMeshEdges);
        _depthProcessingCS.SetFloat("EdgeExtensionOffset", _edgeExtensionOffset);
        _depthProcessingCS.SetFloat("EdgeExtensionDepthOffset", _edgeExtensionDepthOffset);

        // Sets shader resources for the vertex function.
        _depthProcessingCS.SetBuffer(_vertexFromDepthHandle, "vertexBuffer", _vertexBuffer);

        // Sets shader resources for the normal function.
        _depthProcessingCS.SetBuffer(_normalFromVertexHandle, "vertexBuffer", _vertexBuffer);
        _depthProcessingCS.SetBuffer(_normalFromVertexHandle, "normalBuffer", _normalBuffer);
    }

    private void UpdateMesh()
    {
        if (!_initialized)
        {
            return;
        }

        UpdateComputeShaderVariables();

        _depthProcessingCS.Dispatch(_vertexFromDepthHandle, _meshWidth / _numThreadsX,
            (_meshHeight / _numThreadsY) + 1, 1);

        if (_render)
        {
            _vertexBuffer.GetData(_vertices);
            _mesh.vertices = _vertices;
            _mesh.RecalculateNormals();
            _mesh.UploadMeshData(false);
        }
    }

    private void UpdateCollider()
    {
        _vertexBuffer.GetData(_vertices);
        _mesh.vertices = _vertices;
        _meshCollider.sharedMesh = null;
        _meshCollider.sharedMesh = _mesh;
    }

    private void UpdateComputeShaderVariables()
    {
        _depthProcessingCS.SetTexture(_vertexFromDepthHandle, "depthTex", _depthTexture);
        Matrix4x4 screenRotation;
        switch (Screen.orientation)
        {
            case ScreenOrientation.Portrait:
                screenRotation = Matrix4x4.Rotate(Quaternion.Euler(0, 0, -90));
                break;
            case ScreenOrientation.PortraitUpsideDown:
                screenRotation = Matrix4x4.Rotate(Quaternion.Euler(0, 0, 90));
                break;
            case ScreenOrientation.LandscapeRight:
                screenRotation = Matrix4x4.Rotate(Quaternion.Euler(0, 0, 180));
                break;
            default:
                screenRotation = Matrix4x4.Rotate(Quaternion.identity);
                break;
        }
        _depthProcessingCS.SetMatrix("ModelTransform", _cameraManager.transform.localToWorldMatrix * screenRotation);
    }

    private void UpdateRawImage(ref Texture2D texture, XRCpuImage cpuImage)
    {
        if (texture == null || texture.width != cpuImage.width || texture.height != cpuImage.height)
        {
            texture = new Texture2D(cpuImage.width, cpuImage.height, TextureFormat.RGB565, false);
        }

        var conversionParams = new XRCpuImage.ConversionParams(cpuImage, TextureFormat.R16);
        var rawTextureData = texture.GetRawTextureData<byte>();
        cpuImage.Convert(conversionParams, rawTextureData);
        texture.Apply();
    }
}
