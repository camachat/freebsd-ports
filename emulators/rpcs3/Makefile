PORTNAME=	rpcs3
DISTVERSIONPREFIX=	v
DISTVERSION=	0.0.37 # git rev-list --count HEAD
DISTVERSIONSUFFIX=	-17986-g68d2573344
CATEGORIES=	emulators wayland

MAINTAINER=	jbeich@FreeBSD.org
COMMENT=	PS3 emulator/debugger
WWW=		https://rpcs3.net/

LICENSE=	APACHE20 ISCL GPLv2 MIT ZLIB
LICENSE_COMB=	multi
LICENSE_FILE_ISCL=	${WRKSRC_cubeb}/LICENSE
LICENSE_FILE_ZLIB=	${WRKSRC_asmjit}/LICENSE.md

BUILD_DEPENDS=	flatc:devel/flatbuffers
LIB_DEPENDS=	libcurl.so:ftp/curl \
		libpng.so:graphics/png \
		libavcodec.so:multimedia/ffmpeg \
		libpugixml.so:textproc/pugixml

ONLY_FOR_ARCHS=	aarch64 amd64
ONLY_FOR_ARCHS_REASON=	requires int128 and SSE2

USE_GITHUB=	yes
GH_ACCOUNT=	RPCS3
GH_TUPLE=	RPCS3:hidapi:hidapi-0.14.0-53-gf424236:hidapi/3rdparty/hidapi/hidapi \
		RPCS3:soundtouch:2.3.1-78-g3982730:soundtouch/3rdparty/SoundTouch/soundtouch \
		asmjit:asmjit:416f735:asmjit/3rdparty/asmjit/asmjit \
		facebook:zstd:v1.5.7:zstd/3rdparty/zstd/zstd \
		jbeder:yaml-cpp:0.8.0-11-g456c68f:yamlcpp/3rdparty/yaml-cpp/yaml-cpp \
		KhronosGroup:glslang:15.3.0:glslang/3rdparty/glslang/glslang \
		kcat:openal-soft:1.23.1:openal_soft/3rdparty/OpenAL/openal-soft \
		kinetiknz:cubeb:cubeb-0.2-1425-g88585b6:cubeb/3rdparty/cubeb/cubeb \
		Megamouse:VulkanMemoryAllocator:1d8f600:vkalloc/3rdparty/GPUOpen/VulkanMemoryAllocator \
		miniupnp:miniupnp:miniupnpd_2_3_9:miniupnp/3rdparty/miniupnp/miniupnp \
		nothings:stb:013ac3b:stb/3rdparty/stblib/stb \
		wolfSSL:wolfssl:v5.8.0-stable:wolfssl/3rdparty/wolfssl/wolfssl \
		xioTechnologies:Fusion:v1.2.8:fusion/3rdparty/fusion/fusion

USES=		cmake gl iconv:wchar_t llvm:min=19,lib localbase:ldflags \
		openal:soft pkgconfig python:build qt:6 xorg
USE_GL=		gl glew
USE_QT=		base multimedia svg
USE_XORG=	x11
CMAKE_ON=	CMAKE_SKIP_RPATH ${CURL FFMPEG FLATBUFFERS LIBPNG PUGIXML XXHASH:L:S/^/USE_SYSTEM_/}
CMAKE_ON+=	BUNDLE_SPEEX # only used by cubeb tests
CMAKE_OFF=	USE_NATIVE_INSTRUCTIONS USE_PRECOMPILED_HEADERS
LDFLAGS+=	-Wl,--as-needed # GLU

OPTIONS_DEFINE=	ALSA EVDEV FAUDIO JACK LTO OPENCV PULSEAUDIO SDL SNDIO VULKAN
OPTIONS_DEFAULT=EVDEV FAUDIO JACK LTO OPENCV PULSEAUDIO SDL SNDIO VULKAN

ALSA_BUILD_DEPENDS=	alsa-lib>0:audio/alsa-lib
ALSA_CMAKE_BOOL=	USE_ALSA

EVDEV_DESC=		libevdev-based joystick support
EVDEV_CMAKE_BOOL=	USE_LIBEVDEV
EVDEV_BUILD_DEPENDS=	evdev-proto>0:devel/evdev-proto
EVDEV_LIB_DEPENDS=	libevdev.so:devel/libevdev \
			libudev.so:devel/libudev-devd

FAUDIO_DESC=		FAudio audio support
FAUDIO_LIB_DEPENDS=	libFAudio.so:audio/faudio@sdl3
FAUDIO_CMAKE_BOOL=	USE_FAUDIO USE_SYSTEM_FAUDIO

JACK_BUILD_DEPENDS=	jackit>0:audio/jack
JACK_CMAKE_BOOL=	USE_JACK

LTO_CMAKE_BOOL=		USE_LTO

OPENCV_LIB_DEPENDS=	libopencv_imgproc.so:graphics/opencv
OPENCV_CMAKE_BOOL=	USE_SYSTEM_OPENCV

PULSEAUDIO_BUILD_DEPENDS=pulseaudio>0:audio/pulseaudio
PULSEAUDIO_CMAKE_BOOL=	USE_PULSE

SDL_USES=		sdl
SDL_USE=		SDL=sdl3
SDL_CMAKE_BOOL=		USE_SDL USE_SYSTEM_SDL

SNDIO_BUILD_DEPENDS=	sndio>0:audio/sndio
SNDIO_CMAKE_BOOL=	USE_SNDIO

VULKAN_DESC=		Vulkan renderer
VULKAN_BUILD_DEPENDS=	${LOCALBASE}/include/vulkan/vulkan.h:graphics/vulkan-headers
VULKAN_LIB_DEPENDS=	libvulkan.so:graphics/vulkan-loader
VULKAN_CMAKE_BOOL=	USE_VULKAN

# XXX Unbundle audio/rtmidi
GH_TUPLE+=		thestk:rtmidi:6.0.0:rtmidi/3rdparty/rtmidi/rtmidi
ALSA_LIB_DEPENDS+=	libasound.so:audio/alsa-lib
ALSA_CMAKE_BOOL+= 	RTMIDI_API_ALSA
JACK_LIB_DEPENDS+=	libjack.so:audio/jack
JACK_CMAKE_BOOL+=	RTMIDI_API_JACK

post-patch:
	@${REINPLACE_CMD} -e '/GIT_BRANCH/s/local_build/master/' \
		 -e '/GIT_FULL_BRANCH/s/local_build/${GH_ACCOUNT}\/${GH_PROJECT}\/master/' \
		-e '/GIT_VERSION/s/local_build/${GH_TAGNAME:C/[^-]*-//}/' \
		${WRKSRC}/${PORTNAME}/git-version.cmake

post-install:
.if exists(/usr/bin/elfctl)
	${ELFCTL} -e +wxneeded ${STAGEDIR}${PREFIX}/bin/${PORTNAME}
.endif

.include <bsd.port.mk>
