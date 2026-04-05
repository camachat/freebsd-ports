PORTNAME=	ggml
DISTVERSIONPREFIX=	v
DISTVERSION=	0.9.11
CATEGORIES=	misc # machine-learning

MAINTAINER=	yuri@FreeBSD.org
COMMENT=	Tensor library for machine learning
WWW=		https://github.com/ggml-org/ggml

LICENSE=	MIT
LICENSE_FILE=	${WRKSRC}/LICENSE

USES=		cmake:testing compiler:c++17-lang llvm:build,min=22 pkgconfig python:run shebangfix
USE_LDCONFIG=	yes

BROKEN_i386=	compilation fails: LLVM ERROR: out of memory

USE_GITHUB=	yes
GH_ACCOUNT=	ggml-org

CMAKE_ON=	BUILD_SHARED_LIBS \
		GGML_CPU \
		GGML_SSE42 \
		GGML_F16C \
		GGML_FMA \
		GGML_BMI2 \
		GGML_AVX \
		GGML_AVX2
CMAKE_OFF=	GGML_BUILD_TESTS \
		GGML_BUILD_EXAMPLES \
		GGML_AVX512 \
		GGML_AVX512_VBMI \
		GGML_AVX512_VNNI \
		GGML_AVX512_BF16 \
		GGML_AMX_TILE \
		GGML_AMX_INT8 \
		GGML_AMX_BF16 \
		GGML_NATIVE
CMAKE_TESTING_ON=	GGML_BUILD_TESTS

BINARY_ALIAS=	git=false

PLIST_SUB+=	DISTVERSION=${DISTVERSION}

OPTIONS_RADIO=		BLAS
OPTIONS_RADIO_BLAS=	OPENBLAS BLIS
OPTIONS_DEFINE=		VULKAN
OPTIONS_DEFAULT=	VULKAN OPENBLAS
OPTIONS_SUB=		yes

VULKAN_CMAKE_BOOL=	GGML_VULKAN
VULKAN_BUILD_DEPENDS=	glslc:graphics/shaderc \
			vulkan-headers>0:graphics/vulkan-headers
VULKAN_LIB_DEPENDS=	libvulkan.so:graphics/vulkan-loader

OPENBLAS_LIB_DEPENDS=	libopenblas.so:math/openblas
OPENBLAS_CMAKE_ON=	-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS

BLIS_DESC=		BLAS-like libraries from BLIS
BLIS_LIB_DEPENDS=	libblis.so:math/blis
BLIS_CMAKE_ON=		-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=FLAME

.include <bsd.port.options.mk>

.if ${PORT_OPTIONS:MOPENBLAS} || ${PORT_OPTIONS:MBLIS}
PLIST_SUB+=		BLAS=""
SUB_LIST+=		BLAS=""
.else
PLIST_SUB+=		BLAS="@comment "
SUB_LIST+=		BLAS="@comment "
.endif

# tests as of 0.9.5: 100% tests passed, 0 tests failed out of 21

.include <bsd.port.mk>
