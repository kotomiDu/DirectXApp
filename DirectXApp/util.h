#pragma once
#include <stdio.h>
#include <Windows.h>  
#include <d3d11.h>
#include "atlbase.h"
#include "dxgiformat.h"
#define MAX_IN_SURFACES_NUM 2
#define MAX_OUT_SURFACES_NUM 1
static const GUID GUID_VPE_INTERFACE =
{ 0xedd1d4b9, 0x8659, 0x4cbc,{ 0xa4, 0xd6, 0x98, 0x31, 0xa2, 0x16, 0x3a, 0xc3 } };

#define VPE_FN_SEGMENTATION_SET_PARAMS 0x500
typedef enum _VPE_SEGMENTATION_MODE
{
	SEG_MASK = 0,
	SEG_BLENDING_WITH_DEFAULT = 1,
	SEG_BLENDING_WITH_CUSTOME = 2,
}VPE_SEGMENTATION_MODE;
typedef struct _SURFACE_BACKED_BY_SYSMEM
{
	UINT            width;
	UINT            height;
	UINT            size;
	DXGI_FORMAT     format;
	BOOL            bIsBuffer;
	PVOID           pSysMem;
} SURFACE_BACKED_BY_SYSMEM;
typedef struct _VPE_SEGMENTATION_PARAM
{
	BOOL                        bEnableSegmentation;
	VPE_SEGMENTATION_MODE       SegmentationMode;
	SURFACE_BACKED_BY_SYSMEM    Background;
}VPE_SEGMENTATION_PARAM, * PVPE_SEGMENTATION_PARAM;



typedef struct  _SRImageData
{
	BYTE* pbData;                     // pointer to the image data
	UINT     uDataSize;                  // Size of memory
	UINT     uCamIdentifier;             // An identifier to identify camera which took img
	UINT     uSequenceId;                // A sequence number for a camera image
} SRImageData, * pSRImageData;

typedef struct  _SRImage
{
	SRImageData* pImgData;            // pointer to the image data structure
	UINT                       uWidth;               // image width ¨C width:height=2:1 for 360:180 view
	UINT                       uHeight;              // image height
	INT                        uPitch;               // image pitch as image is stored
	DXGI_FORMAT                uFormat;              // pixel format ¨C Pixel layout
	UINT                       uColorStd;            // Future - RGB color standard BT2020/BT709
	INT                        uBitDepth;            // Future - Bit depth of each color component
	BYTE* pbAlphaMap;          // pointer to Alpha map
	UINT                       uPrivateDriverDataSize;  // Private Data used by SDK/Driver
	VOID* pvPrivateDriverData;     // Private Data used by SDK/Driver
	UINT                       uPrivateDataSize;     // Future - Extension to this structure
	VOID* pvPrivateData;        // Future - Extension to this structure
	_SRImage* next;                 // pointer to next image
}  SRImage, * pSRImage;

typedef struct  _SRHeapImage
{
	UINT              uNumOfStreams;       // number of stream/cameras pInputFile Rig
	SRImage* pbImgHeapHead;      // pointer to the image data structure
}  SRHeapImage, * pSRHeapImage;

enum FrameRateControl
{
	DX11_FRC_NOOP = 0,
	DX11_FRC_NORMAL,
	DX11_FRC_24To60,
	DX11_FRC_30To60,
	DX11_FRC_COUNT
};

enum ScalingMode
{
	SCALING_MODE_DEFAULT = 0,                 // Default
	SCALING_MODE_QUALITY,                     // LowerPower
	SCALING_MODE_SUPERRESOLUTION              // SuperREsolution
};

enum VPEMode
{
	VPE_MODE_NONE = 0x0,
	VPE_MODE_PREPROC = 0x1,
};

enum VPE_VERSION_ENUM
{
	VPE_VERSION_1_0 = 0x0001,
	VPE_VERSION_2_0 = 0x0002,
	VPE_VERSION_3_0 = 0x0003,   // CNL new campipe interface
	VPE_VERSION_UNKNOWN = 0xffff,
};

typedef struct _SR_SCALING_MODE
{
	UINT Fastscaling;
	// to be extention
	// where customer can pass the training model data to DXVA driver
}SR_SCALING_MODE, * PSR_SCALING_MODE;

typedef struct _VPE_VERSION
{
	UINT Version;
}VPE_VERSION, * PVPE_VERSION;

typedef struct _VPE_MODE
{
	UINT Mode;
}VPE_MODE, * PVPE_MODE;

typedef struct _VPE_FUNCTION
{
	UINT                                        Function;               // [pInputFile]
	union                                                               // [pInputFile]
	{
		void* pSegmentationParam;
		void* pVPEMode;
		void* pVPEVersion;
	};
} VPE_FUNCTION, * PVPE_FUNCTION;

class CDX11VPExtension
{
public:
	CDX11VPExtension(
		ID3D11Device* pd3d11Device,
		ID3D11DeviceContext* pd3d11DeviceContext,
		ID3D11VideoProcessor* pd3d11VideoProcessor,
		ID3D11VideoContext* pd3d11VideoContext);
	~CDX11VPExtension();

	HRESULT QueryVPEGuids();
	HRESULT QueryVPECaps();

	HRESULT VideoProcessorSetVPreMode();

private:
	CComPtr<ID3D11DeviceContext>                m_pD3D11DeviceContext;
	CComPtr<ID3D11VideoProcessor>               m_pD3D11VideoProcessor;
	CComPtr<ID3D11VideoContext>                 m_pD3D11VideoContext;
	CComPtr<ID3D11Device>                       m_pD3D11Device;
	UINT                                        m_uiVPExtGuidCount;

	VPE_VERSION                                 m_VPEVersion = {};
	VPE_SEGMENTATION_PARAM                      m_Segmentation = {};
};

class VidoeProcessor
{
public:
	VidoeProcessor();
	virtual ~VidoeProcessor()
	{

	}

	HRESULT CreateDX11Device();
	HRESULT DestroyDX11Device();

	HRESULT CreateDX11VideoProcessor(D3D11_VIDEO_PROCESSOR_CONTENT_DESC* pDesc,
		D3D11_VIDEO_USAGE                   DeviceUsage,
		FrameRateControl                    eFrc);

	HRESULT CheckVPInputFormat(DXGI_FORMAT inputFormat);
	HRESULT CheckVPOutputFormat(DXGI_FORMAT outputFormat);

	HRESULT CreateInputView(ID3D11Resource* pRes,
		D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC* pDesc,
		ID3D11VideoProcessorInputView** ppInputView);

	HRESULT CreateOutputView(ID3D11Resource* pRes,
		D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC* pDesc,
		ID3D11VideoProcessorOutputView** ppOutputView);

	HRESULT VideoProcessorBlt(pSRImage                   pInputImgs,
		pSRImage                       pOutputImg);

	HRESULT VideoProcessorBlt(ID3D11Texture2D* pInputSuf,
		ID3D11Texture2D* pOutputSuf);

	HRESULT CopyFromCPUToGPU(pSRImage pCurImg);
	HRESULT CopyFromGPUToCPU(pSRImage pCurImg);

	HRESULT AllocateSurface(SRImage* pImage, UINT uWidth, UINT uHeight, DXGI_FORMAT fourCC, BOOL bInput);
	HRESULT FreeSurface(SRImage* pImage);

public:
	CComPtr<ID3D11Device>                               m_pD3D11Device = nullptr;
	CComPtr<ID3D11DeviceContext>                        m_pD3D11DeviceContext = nullptr;
	CComPtr<ID3D11VideoDevice>                          m_pD3D11VideoDevice = nullptr;
	CComPtr<ID3D11VideoContext>                         m_pD3D11VideoContext = nullptr;
	CComPtr<ID3D11VideoProcessorEnumerator>             m_pD3D11VideoProcessorEnum = nullptr;
	CComPtr<ID3D11VideoProcessor>                       m_pD3D11VideoProcessor = nullptr;

	CDX11VPExtension* m_pDX11VPExtension;
	D3D11_VIDEO_PROCESSOR_STREAM                        m_InStreams[MAX_IN_SURFACES_NUM];
};