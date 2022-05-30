#include <stdio.h>
#include <Windows.h>  
#include <iostream> 
#include <d3d11.h>
#include "atlbase.h"
#include "dxgiformat.h"
#include "util.h"
#define CHK_HR(hr)		{ if(FAILED(hr))		return hr;}
#define CHK_NULL(ptr)	{ if(!ptr)				return hr;}

#define VPE_FN_SCALING_MODE_PARAM            0x37
#define VPE_FN_MODE_PARAM                    0x20
#define VPE_FN_SET_VERSION_PARAM             0x01

#define DXGI_FORMAT_MAX_COUNT 150

const int uBytesPerPixel = 4;

VidoeProcessor::VidoeProcessor()
{
}

HRESULT VidoeProcessor::CreateDX11Device()
{
	HRESULT hr = E_FAIL;
	D3D_FEATURE_LEVEL MaxSupportedFeatureLevel = D3D_FEATURE_LEVEL_9_1;

	D3D_FEATURE_LEVEL FeatureLevels[] = {
		D3D_FEATURE_LEVEL_11_1,
		D3D_FEATURE_LEVEL_11_0,
		D3D_FEATURE_LEVEL_10_1,
		D3D_FEATURE_LEVEL_10_0,
		D3D_FEATURE_LEVEL_9_3,
		D3D_FEATURE_LEVEL_9_2,
		D3D_FEATURE_LEVEL_9_1
	};

	/*hr = D3D11CreateDevice(
		NULL,
		D3D_DRIVER_TYPE_HARDWARE,
		NULL,
		D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
		FeatureLevels,
		ARRAYSIZE(FeatureLevels),
		D3D11_SDK_VERSION,
		&m_pD3D11Device,
		&MaxSupportedFeatureLevel,
		&m_pD3D11DeviceContext);

	CHK_HR(hr);*/

	hr = m_pD3D11Device->QueryInterface(__uuidof(ID3D11VideoDevice), (void**)&m_pD3D11VideoDevice);
	CHK_HR(hr);

	hr = m_pD3D11DeviceContext->QueryInterface(__uuidof(ID3D11VideoContext), (void**)&m_pD3D11VideoContext);
	CHK_HR(hr);

	hr = CreateDX11VideoProcessor(NULL, D3D11_VIDEO_USAGE_PLAYBACK_NORMAL, DX11_FRC_NORMAL);
	CHK_HR(hr);

	// Create DX11 VPE
	m_pDX11VPExtension = new CDX11VPExtension(m_pD3D11Device, m_pD3D11DeviceContext, m_pD3D11VideoProcessor, m_pD3D11VideoContext);
	CHK_NULL(m_pDX11VPExtension);


	return hr;
}

HRESULT VidoeProcessor::DestroyDX11Device()
{
	HRESULT hr = E_FAIL;
	return hr;
}

HRESULT VidoeProcessor::CreateDX11VideoProcessor(
	D3D11_VIDEO_PROCESSOR_CONTENT_DESC* pDesc,
	D3D11_VIDEO_USAGE                    DeviceUsage,
	FrameRateControl                     eFrc)
{
	HRESULT hr = S_OK;
	D3D11_VIDEO_PROCESSOR_CONTENT_DESC desc, * pVideoDesc = NULL;

	if (pDesc == NULL)
	{
		// video desc is only a hint for device creation 
		ZeroMemory(&desc, sizeof(D3D11_VIDEO_PROCESSOR_CONTENT_DESC));
		desc.InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;

		desc.InputFrameRate.Numerator = 30;
		desc.InputFrameRate.Denominator = 1;
		desc.InputHeight = 540;
		desc.InputWidth = 960;
		desc.OutputHeight = 1080;
		desc.OutputWidth = 1920;

		switch (eFrc)
		{
		case DX11_FRC_24To60:
			desc.OutputFrameRate.Numerator = 60;
			desc.OutputFrameRate.Denominator = 24;
			break;
		case DX11_FRC_30To60:
			desc.OutputFrameRate.Numerator = 60;
			desc.OutputFrameRate.Denominator = 30;
			break;
		case DX11_FRC_NOOP:
		case DX11_FRC_NORMAL:
		default:
			desc.OutputFrameRate.Numerator = 1;
			desc.OutputFrameRate.Denominator = 1;
			break;
		}
		pVideoDesc = &desc;
	}
	else
	{
		pVideoDesc = pDesc;
	}

	// Create the DXVA enumerator
	hr = m_pD3D11VideoDevice->CreateVideoProcessorEnumerator(pVideoDesc, &m_pD3D11VideoProcessorEnum);
	CHK_HR(hr);

	// Get the DXVP device caps.
	D3D11_VIDEO_PROCESSOR_CAPS caps;
	ZeroMemory(&caps, sizeof(caps));
	hr = m_pD3D11VideoProcessorEnum->GetVideoProcessorCaps(&caps);
	CHK_HR(hr);

	UINT formatFlags = 0;
	for (UINT i = 0; i < DXGI_FORMAT_MAX_COUNT; i++)
	{
		hr = m_pD3D11VideoProcessorEnum->CheckVideoProcessorFormat((DXGI_FORMAT)i, &formatFlags);
		CHK_HR(hr);
	}

	// Create DX11 video processor 
	hr = m_pD3D11VideoDevice->CreateVideoProcessor(m_pD3D11VideoProcessorEnum, 0/*Frame rate*/, &m_pD3D11VideoProcessor);
	CHK_HR(hr);

	return hr;

}

HRESULT VidoeProcessor::CheckVPInputFormat(DXGI_FORMAT inputFormat)
{
	HRESULT hr = E_FAIL;
	UINT formatFlags = 0;

	hr = m_pD3D11VideoProcessorEnum->CheckVideoProcessorFormat(inputFormat, &formatFlags);
	CHK_HR(hr);

	return hr;
}

HRESULT VidoeProcessor::CheckVPOutputFormat(DXGI_FORMAT outputFormat)
{
	HRESULT hr = E_FAIL;
	UINT formatFlags = 0;

	hr = m_pD3D11VideoProcessorEnum->CheckVideoProcessorFormat(outputFormat, &formatFlags);
	CHK_HR(hr);

	return hr;
}

HRESULT VidoeProcessor::CreateInputView(ID3D11Resource* pRes, D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC* pDesc, ID3D11VideoProcessorInputView** ppInputView)
{
	HRESULT hr = S_OK;
	hr = m_pD3D11VideoDevice->CreateVideoProcessorInputView(pRes, m_pD3D11VideoProcessorEnum, pDesc, ppInputView);

	return hr;
}

HRESULT VidoeProcessor::CreateOutputView(ID3D11Resource* pRes, D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC* pDesc, ID3D11VideoProcessorOutputView** ppOutputView)
{
	HRESULT hr = E_FAIL;
	ID3D11VideoProcessorOutputView* pOutputView = NULL;
	hr = m_pD3D11VideoDevice->CreateVideoProcessorOutputView(pRes, m_pD3D11VideoProcessorEnum, pDesc, &pOutputView);
	CHK_HR(hr);

	*ppOutputView = pOutputView;
	return hr;
}
HRESULT VidoeProcessor::VideoProcessorBlt(ID3D11Texture2D* pInputSuf, ID3D11Texture2D* pOutputSuf)
{
	HRESULT hr = E_FAIL;
	for (UINT uNumOfStreams = 0; uNumOfStreams < 1; )
	{

		m_InStreams[uNumOfStreams].Enable = TRUE;
		m_InStreams[uNumOfStreams].InputFrameOrField = 0;
		m_InStreams[uNumOfStreams].OutputIndex = 0;
		m_InStreams[uNumOfStreams].ppFutureSurfaces = nullptr;
		m_InStreams[uNumOfStreams].ppPastSurfaces = nullptr;
		m_InStreams[uNumOfStreams].ppFutureSurfacesRight = nullptr;
		m_InStreams[uNumOfStreams].ppPastSurfacesRight = nullptr;
		m_InStreams[uNumOfStreams].FutureFrames = 0;
		m_InStreams[uNumOfStreams].PastFrames = 0;
		m_InStreams[uNumOfStreams].pInputSurfaceRight = nullptr;
		m_InStreams[uNumOfStreams].pInputSurface = nullptr;

		D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC InputViewDesc;
		memset((void*)&InputViewDesc, 0, sizeof(InputViewDesc));
		InputViewDesc.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
		hr = CreateInputView(pInputSuf, &InputViewDesc, &m_InStreams[uNumOfStreams].pInputSurface);
		CHK_HR(hr);
		uNumOfStreams++;
	}
	
	// Create output view 
	CComPtr<ID3D11VideoProcessorOutputView>         pOutView;
	D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC          outputViewDesc;
	memset((void*)&outputViewDesc, 0, sizeof(outputViewDesc));
	outputViewDesc.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;

	hr = CreateOutputView(pOutputSuf, &outputViewDesc, &pOutView);

	if (SUCCEEDED(hr))
	{

		hr = m_pD3D11VideoContext->VideoProcessorBlt(m_pD3D11VideoProcessor, pOutView, 0, 1, m_InStreams);

		// TODO: if system memory needed, call GPU result to CPU.
		//CopyFromGPUToCPU(pOutputImg);
	}

	return hr;

}
HRESULT VidoeProcessor::VideoProcessorBlt(pSRImage pInputImgs, pSRImage pOutputImg)
{
	HRESULT hr = E_FAIL;

	pSRImage  pCurrentImage = pInputImgs;

	// Copy Input to surface
	for (UINT uNumOfStreams = 0; uNumOfStreams < 1; )
	{
		CopyFromCPUToGPU(pCurrentImage);

		m_InStreams[uNumOfStreams].Enable = TRUE;
		m_InStreams[uNumOfStreams].InputFrameOrField = 0;
		m_InStreams[uNumOfStreams].OutputIndex = 0;
		m_InStreams[uNumOfStreams].ppFutureSurfaces = nullptr;
		m_InStreams[uNumOfStreams].ppPastSurfaces = nullptr;
		m_InStreams[uNumOfStreams].ppFutureSurfacesRight = nullptr;
		m_InStreams[uNumOfStreams].ppPastSurfacesRight = nullptr;
		m_InStreams[uNumOfStreams].FutureFrames = 0;
		m_InStreams[uNumOfStreams].PastFrames = 0;
		m_InStreams[uNumOfStreams].pInputSurfaceRight = nullptr;
		m_InStreams[uNumOfStreams].pInputSurface = nullptr;

		D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC InputViewDesc;
		memset((void*)&InputViewDesc, 0, sizeof(InputViewDesc));
		InputViewDesc.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
		hr = CreateInputView((ID3D11Texture2D*)(pCurrentImage->pvPrivateDriverData), &InputViewDesc, &m_InStreams[uNumOfStreams].pInputSurface);
		CHK_HR(hr);
		pCurrentImage = pCurrentImage->next;
		uNumOfStreams++;
	}

	// Create output view 
	CComPtr<ID3D11VideoProcessorOutputView>         pOutView;
	D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC          outputViewDesc;
	memset((void*)&outputViewDesc, 0, sizeof(outputViewDesc));
	outputViewDesc.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;

	hr = CreateOutputView((ID3D11Texture2D*)(pOutputImg->pvPrivateDriverData), &outputViewDesc, &pOutView);
	CHK_HR(hr);
	if (SUCCEEDED(hr))
	{

		hr = m_pD3D11VideoContext->VideoProcessorBlt(m_pD3D11VideoProcessor, pOutView, 0, 1, m_InStreams);

		// TODO: if system memory needed, call GPU result to CPU.
		CopyFromGPUToCPU(pOutputImg);
	}

	return hr;
}

HRESULT VidoeProcessor::CopyFromCPUToGPU(pSRImage pCurImage)
{
	HRESULT hr = E_FAIL;

	ID3D11Texture2D* pInputStage = (ID3D11Texture2D*)(pCurImage->pvPrivateData);
	ID3D11Texture2D* pInputSurf = (ID3D11Texture2D*)(pCurImage->pvPrivateDriverData);

	D3D11_MAPPED_SUBRESOURCE lr = { 0 };
	hr = m_pD3D11DeviceContext->Map(pInputStage, 0, D3D11_MAP_WRITE, 0, &lr);
	UINT nWidth = pCurImage->uWidth;
	UINT nHeight = pCurImage->uHeight;
	UINT dxFormat = pCurImage->uFormat;

	BYTE* pTemp = (BYTE*)lr.pData;
	BYTE* pIn = pCurImage->pImgData->pbData;
	BYTE* pOut = pTemp;
	if (pCurImage->uFormat == DXGI_FORMAT_NV12)
	{
		for (UINT iRow = 0; iRow < nHeight * 3 / 2; iRow++)
		{
			CopyMemory(pOut, pIn, nWidth);
			pOut += lr.RowPitch;
			pIn += pCurImage->uWidth;
		}
	}
	else if (pCurImage->uFormat == DXGI_FORMAT_R8G8_UNORM)
	{
		for (UINT iRow = 0; iRow < nHeight / 2; iRow++)
		{
			CopyMemory(pOut, pIn, nWidth);
			pOut += lr.RowPitch;
			pIn += pCurImage->uWidth;
		}
	}
	else if (pCurImage->uFormat == DXGI_FORMAT_B8G8R8A8_UNORM)
	{
		for (UINT iRow = 0; iRow < nHeight; iRow++)
		{
			CopyMemory(pOut, pIn, uBytesPerPixel * nWidth);
			pOut += lr.RowPitch;
			pIn += pCurImage->uWidth * uBytesPerPixel;
		}
	}
	else
	{
		for (UINT iRow = 0; iRow < nHeight; iRow++)
		{
			CopyMemory(pOut, pIn, uBytesPerPixel * nWidth);
			pOut += lr.RowPitch;
			pIn += pCurImage->uWidth * 4;
		}
	}

	m_pD3D11DeviceContext->Unmap(pInputStage, 0);
	m_pD3D11DeviceContext->CopyResource(pInputSurf, pInputStage);

	return hr;
}

HRESULT VidoeProcessor::CopyFromGPUToCPU(pSRImage pCurImg)
{
	HRESULT hr = E_FAIL;

	ID3D11Texture2D* pOutputStage = (ID3D11Texture2D*)(pCurImg->pvPrivateData);
	ID3D11Texture2D* pOutputSurf = (ID3D11Texture2D*)(pCurImg->pvPrivateDriverData);

	m_pD3D11DeviceContext->CopyResource(pOutputStage, pOutputSurf);

	D3D11_MAPPED_SUBRESOURCE lr = { 0 };
	hr = m_pD3D11DeviceContext->Map(pOutputStage, 0, D3D11_MAP_READ, 0, &lr);

	BYTE* pOut = (BYTE*)lr.pData;
	UINT uPitch = lr.RowPitch;
	BYTE* pTemp = pOut;
	BYTE* pDest = pCurImg->pImgData->pbData;

	if (pCurImg->uFormat == DXGI_FORMAT_NV12)
	{
		for (size_t i = 0; i < pCurImg->uHeight * 3 / 2; i++)
		{
			memcpy(pDest, pOut, pCurImg->uWidth);
			pOut += lr.RowPitch;
			pDest += pCurImg->uWidth;
		}
	}
	else if (pCurImg->uFormat == DXGI_FORMAT_B8G8R8A8_UNORM)
	{
		for (size_t i = 0; i < pCurImg->uHeight; i++)
		{
			memcpy(pDest, pOut, pCurImg->uWidth * uBytesPerPixel);
			pOut += lr.RowPitch;
			pDest += pCurImg->uWidth * uBytesPerPixel;
		}
	}
	else
	{
		for (size_t i = 0; i < pCurImg->uHeight; i++)
		{
			memcpy(pDest, pOut, pCurImg->uWidth * uBytesPerPixel);
			pOut += lr.RowPitch;
			pDest += pCurImg->uWidth * uBytesPerPixel;
		}
	}

	return hr;
}

HRESULT VidoeProcessor::AllocateSurface(SRImage* pImage, UINT uWidth, UINT uHeight, DXGI_FORMAT fourCC, BOOL bInput)
{
	HRESULT hr = S_OK;

	D3D11_TEXTURE2D_DESC descTexture = { 0 };
	descTexture.Width = uWidth;
	descTexture.Height = uHeight;
	descTexture.MipLevels = 1;
	descTexture.ArraySize = 1;
	descTexture.Format = fourCC;
	descTexture.SampleDesc.Count = 1;

	if (pImage->pvPrivateData || pImage->pvPrivateDriverData)
	{
		return hr; // already allocated a surface
	}

	if (bInput)
	{

		descTexture.Usage = D3D11_USAGE_STAGING;
		descTexture.BindFlags = 0;
		descTexture.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		hr = m_pD3D11Device->CreateTexture2D(&descTexture, NULL, (ID3D11Texture2D**)(&pImage->pvPrivateData));
		descTexture.Usage = D3D11_USAGE_DEFAULT;
		descTexture.BindFlags = 0;
		descTexture.CPUAccessFlags = 0;
		hr = m_pD3D11Device->CreateTexture2D(&descTexture, NULL, (ID3D11Texture2D**)(&pImage->pvPrivateDriverData));
	}
	else  //Output
	{
		descTexture.Usage = D3D11_USAGE_STAGING;
		descTexture.BindFlags = 0;
		descTexture.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		hr = m_pD3D11Device->CreateTexture2D(&descTexture, NULL, (ID3D11Texture2D**)(&pImage->pvPrivateData));
		descTexture.Usage = D3D11_USAGE_DEFAULT;
		descTexture.BindFlags = D3D11_BIND_RENDER_TARGET;
		descTexture.CPUAccessFlags = 0;
		hr = m_pD3D11Device->CreateTexture2D(&descTexture, NULL, (ID3D11Texture2D**)(&pImage->pvPrivateDriverData));
	}

	return hr;
}

HRESULT VidoeProcessor::FreeSurface(SRImage* pImage)
{
	ID3D11Texture2D* pTexture = (ID3D11Texture2D*)(pImage->pvPrivateDriverData);
	if (pTexture)
	{
		pTexture->Release();
		pTexture = nullptr;
	}
	ID3D11Texture2D* pTexturePrivate = (ID3D11Texture2D*)(pImage->pvPrivateData);
	if (pTexturePrivate)
	{
		pTexturePrivate->Release();
		pTexturePrivate = nullptr;
	}
	return S_OK;
}

CDX11VPExtension::CDX11VPExtension(
	ID3D11Device* pd3d11Device,
	ID3D11DeviceContext* pd3d11DeviceContext,
	ID3D11VideoProcessor* pd3d11VideoProcessor,
	ID3D11VideoContext* pd3d11VideoContext) :
	m_pD3D11Device(pd3d11Device),
	m_pD3D11DeviceContext(pd3d11DeviceContext),
	m_pD3D11VideoProcessor(pd3d11VideoProcessor),
	m_pD3D11VideoContext(pd3d11VideoContext)
{
	VideoProcessorSetVPreMode();
}

CDX11VPExtension::~CDX11VPExtension()
{
}

HRESULT CDX11VPExtension::QueryVPEGuids()
{
	return S_OK;
}

HRESULT CDX11VPExtension::QueryVPECaps()
{
	return S_OK;
}

HRESULT CDX11VPExtension::VideoProcessorSetVPreMode()
{
	HRESULT      hr = E_FAIL;
	const GUID* pExtensionGuid = nullptr;
	UINT         DataSize = 0;
	void* pData = nullptr;

	VPE_FUNCTION functionParams;

	// Set VPE Version
	memset((PVOID)&functionParams, 0, sizeof(functionParams));

	m_VPEVersion.Version = (UINT)VPE_VERSION_3_0;
	functionParams.Function = VPE_FN_SET_VERSION_PARAM;
	functionParams.pVPEVersion = &m_VPEVersion;

	pData = &functionParams;
	DataSize = sizeof(functionParams);
	pExtensionGuid = &GUID_VPE_INTERFACE;

	hr = m_pD3D11VideoContext->VideoProcessorSetOutputExtension(m_pD3D11VideoProcessor, pExtensionGuid, DataSize, pData);

	// Set VPE Mode
	memset((PVOID)&functionParams, 0, sizeof(functionParams));

	m_Segmentation.bEnableSegmentation = true;
	m_Segmentation.SegmentationMode = SEG_BLENDING_WITH_DEFAULT;
	functionParams.Function = VPE_FN_SEGMENTATION_SET_PARAMS;
	functionParams.pSegmentationParam = &m_Segmentation;

	pData = &functionParams;
	DataSize = sizeof(functionParams);
	pExtensionGuid = &GUID_VPE_INTERFACE;

	hr = m_pD3D11VideoContext->VideoProcessorSetOutputExtension(m_pD3D11VideoProcessor, pExtensionGuid, DataSize, pData);

	return hr;
}



int usage()
{
	fprintf(stderr, "How to use: TEST_SR -i <input file> -o <output file> -if <input format> -of <output format> \n"   \
		"-iw <input width> -ih <input height> -ow <output width> -oh <output height> \n");
	return -1;
}

int main1(int argc, char** argv)
{
	if (argc < 17)
	{
		usage();
	}


	// get input parameters
	char inputfileName[260], outputfileName[260];
	DXGI_FORMAT inputformat, outputformat;
	uint32_t inputWidth = 0, inputHeight = 0, outputWidth = 0, outputHeight = 0;

	for (int i = 1; i < argc; i++)
	{
		if (argv[i] == nullptr)
		{
			return -1;
		}
		if (!strcmp(argv[i], "-i"))
		{
			if (strlen(argv[i + 1]) < 260)
			{
				strcpy_s(inputfileName, argv[i + 1]);
			}
			else
			{
				fprintf(stderr, "Input file name is too long! \n");
				return -1;
			}
			i++;
		}
		else if (!strcmp(argv[i], "-o"))
		{
			if (strlen(argv[i + 1]) < 260)
			{
				strcpy_s(outputfileName, argv[i + 1]);
			}
			else
			{
				fprintf(stderr, "Output file name is too long! \n");
				return -1;
			}
			i++;
		}
		else if (!strcmp(argv[i], "-if"))
		{
			if (!strcmp(argv[i + 1], "nv12") || !strcmp(argv[i + 1], "NV12"))
			{
				inputformat = DXGI_FORMAT_NV12;
			}
			else if (!strcmp(argv[i + 1], "argb") || !strcmp(argv[i + 1], "ARGB"))
			{
				inputformat = DXGI_FORMAT_B8G8R8A8_UNORM;
			}
			else
			{
				fprintf(stderr, "Unsupported format!");
				return -1;
			}
			i++;
		}
		else if (!strcmp(argv[i], "-of"))
		{
			if (!strcmp(argv[i + 1], "nv12") || !strcmp(argv[i + 1], "NV12"))
			{
				outputformat = DXGI_FORMAT_NV12;
			}
			else if (!strcmp(argv[i + 1], "argb") || !strcmp(argv[i + 1], "ARGB"))
			{
				outputformat = DXGI_FORMAT_B8G8R8A8_UNORM;
			}
			else
			{
				fprintf(stderr, "Unsupported format!");
				return -1;
			}
			i++;
		}
		else if (!strcmp(argv[i], "-iw"))
		{
			sscanf_s(argv[i + 1], "%u", &inputWidth);
			i++;
		}
		else if (!strcmp(argv[i], "-ih"))
		{
			sscanf_s(argv[i + 1], "%u", &inputHeight);
			i++;
		}
		else if (!strcmp(argv[i], "-ow"))
		{
			sscanf_s(argv[i + 1], "%u", &outputWidth);
			i++;
		}
		else if (!strcmp(argv[i], "-oh"))
		{
			sscanf_s(argv[i + 1], "%u", &outputHeight);
			i++;
		}

	}
	FILE* pInputFile = nullptr, * pOutputFile = nullptr;
	fopen_s(&pInputFile, inputfileName, "rb");

	if (!pInputFile)
	{
		fprintf(stderr, "Can't open input file!");
		return -1;
	}

	fopen_s(&pOutputFile, outputfileName, "wb");
	if (!pOutputFile)
	{
		fprintf(stderr, "Can't create output file!");
		return -1;
	}

	SRImage inputImg;
	SRImage outputImg;
	uint32_t inSize = 0;
	// input size calculation
	switch (inputformat)
	{
	case DXGI_FORMAT_NV12: inSize = inputWidth * inputHeight * 3 / 2; break;
	case DXGI_FORMAT_B8G8R8A8_UNORM: inSize = inputWidth * inputHeight * 4; break;
	default: inSize = inputWidth * inputHeight * 3 / 2;
	}

	memset(&inputImg, 0, sizeof(SRImage));
	memset(&outputImg, 0, sizeof(SRImage));

	HRESULT hr = S_OK;
	VidoeProcessor* pDX11VideoProcessor = new VidoeProcessor();

	if (pDX11VideoProcessor)
	{
		hr = pDX11VideoProcessor->CreateDX11Device();
	}
	else
	{
		return E_FAIL;
	}
	//----------------------------------------
	// Allocate buffers for input/output
	//----------------------------------------

	if (FAILED(hr))
	{
		fprintf(stderr, "Initialize failed!\n");
		return -1;
	}

	// Allocate input buffer
	inputImg.pImgData = new SRImageData;
	inputImg.pImgData->pbData = (unsigned char*)malloc(inSize);

	if (!inputImg.pImgData->pbData)
	{
		fprintf(stderr, "Allocate memory fail!\n");
		return -1;
	}

	inputImg.pbAlphaMap = nullptr;
	inputImg.uHeight = inputHeight;
	inputImg.uWidth = inputWidth;
	inputImg.uPitch = inputWidth;
	inputImg.uFormat = inputformat;
	inputImg.uBitDepth = 8;

	hr = pDX11VideoProcessor->AllocateSurface(&inputImg, inputWidth, inputHeight, inputformat, true);

	if (FAILED(hr))
	{
		fprintf(stderr, "AllocImageMem failed!\n");
		return -1;
	}

	// Allocate Output buffer
	int outsize = 0;
	switch (outputformat)
	{
	case DXGI_FORMAT_NV12: outsize = outputWidth * outputHeight * 3 / 2; break;
	case DXGI_FORMAT_B8G8R8A8_UNORM: outsize = outputWidth * outputHeight * 4; break;
	default: outsize = outputWidth * outputHeight * 3 / 2;
	}

	outputImg.pImgData = new SRImageData;
	outputImg.pImgData->pbData = (unsigned char*)malloc(outsize);

	if (!outputImg.pImgData->pbData)
	{
		fprintf(stderr, "Allocate memory fail!\n");
		return -1;
	}

	outputImg.pbAlphaMap = nullptr;
	outputImg.uHeight = outputHeight;
	outputImg.uWidth = outputWidth;
	outputImg.uPitch = outputWidth;
	outputImg.uFormat = outputformat;
	outputImg.uBitDepth = 8;

	hr = pDX11VideoProcessor->AllocateSurface(&outputImg, outputWidth, outputHeight, outputformat, false);
	if (FAILED(hr))
	{
		fprintf(stderr, "AllocImageMem failed!\n");
		return -1;
	}

	//----------------------------------------
	// prepare input data
	//----------------------------------------
	if (!fread(inputImg.pImgData->pbData, inSize, 1, pInputFile))
	{
		printf("Fail to read input file!");
		return -1;
	}
	fclose(pInputFile);
	//----------------------------------------
	// SEG Render
	//----------------------------------------
	//for (int i = 0; i < 1000; i++)
	{
		hr = pDX11VideoProcessor->VideoProcessorBlt(&inputImg, &outputImg);
	}

	if (FAILED(hr))
	{
		fprintf(stderr, "Render failed!\n");
		return -1;
	}

	// Write output data
	if (!fwrite(outputImg.pImgData->pbData, outsize, 1, pOutputFile))
	{
		printf("Fail to write output file!");
		return -1;
	}
	fclose(pOutputFile);
	//
	 //----------------------------------------
	 // Free resource
	 //----------------------------------------

	hr = pDX11VideoProcessor->FreeSurface(&inputImg);
	if (FAILED(hr))
	{
		fprintf(stderr, "DestroyImageMem failed!\n");
		return -1;
	}

	hr = pDX11VideoProcessor->FreeSurface(&outputImg);
	if (FAILED(hr))
	{
		fprintf(stderr, "DestroyImageMem failed!\n");
		return -1;
	}

	if (pDX11VideoProcessor)
	{
		pDX11VideoProcessor->DestroyDX11Device();
		delete pDX11VideoProcessor;
		pDX11VideoProcessor = nullptr;
	}

	if (inputImg.pImgData)
	{
		free(inputImg.pImgData->pbData);
		inputImg.pImgData->pbData = nullptr;

		delete inputImg.pImgData;
		inputImg.pImgData = nullptr;
	}

	if (outputImg.pImgData)
	{
		free(outputImg.pImgData->pbData);
		outputImg.pImgData->pbData = nullptr;

		delete outputImg.pImgData;
		outputImg.pImgData = nullptr;
	}

}