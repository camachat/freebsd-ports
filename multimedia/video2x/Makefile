PORTNAME=	video2x
DISTVERSION=	6.4.0
CATEGORIES=	multimedia

MAINTAINER=	eric@camachat.org
COMMENT=	A machine learning-based video super resolution and frame interpolation framework.
WWW=		https://github.com/k4yt3x/video2x

LICENSE=	AGPLv3

BUILD_DEPENDS=	boost-libs>0:devel/boost-libs \
		glslang>0:graphics/glslang \
		ncnn>0:misc/ncnn \
		vulkan-headers>0:graphics/vulkan-headers
LIB_DEPENDS=	libavcodec.so:multimedia/ffmpeg \
		libspdlog.so:devel/spdlog \
		libvulkan.so:graphics/vulkan-loader

USES=		cmake compiler:c++17-lang pkgconfig

USE_GITHUB=	yes
GH_ACCOUNT=	k4yt3x
GH_TUPLE=	k4yt3x:librealcugan-ncnn-vulkan:d9c5a7e:realcugan/third_party/librealcugan_ncnn_vulkan \
		k4yt3x:librealesrgan-ncnn-vulkan:c1f2555:realesrgan/third_party/librealesrgan_ncnn_vulkan \
		k4yt3x:librife-ncnn-vulkan:f1de3a9:rife/third_party/librife_ncnn_vulkan

CMAKE_ON=	VIDEO2X_USE_EXTERNAL_BOOST VIDEO2X_USE_EXTERNAL_NCNN VIDEO2X_USE_EXTERNAL_SPDLOG

OPTIONS_DEFINE=		NATIVE
OPTIONS_DEFAULT=

OPTIONS_DEFINE_amd64=		AVX2 AVX512
OPTIONS_DEFAULT_amd64=		AVX2 AVX512

NATIVE_CMAKE_BOOL=	VIDEO2X_ENABLE_NATIVE

AVX2_DESC=	Enable x86-64-v3 (AVX2) optimizations
AVX2_CMAKE_BOOL=	VIDEO2X_ENABLE_X86_64_V3

AVX512_DESC=	Enable x86-64-v4 (AVX-512) optimizations
AVX512_CMAKE_BOOL=	VIDEO2X_ENABLE_X86_64_V4

.include <bsd.port.mk>
