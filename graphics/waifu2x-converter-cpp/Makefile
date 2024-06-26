PORTNAME=	waifu2x-converter-cpp
DISTVERSIONPREFIX=	v
DISTVERSION=	5.3.4
PORTREVISION=	5
CATEGORIES=	graphics

MAINTAINER=	ports@FreeBSD.org
COMMENT=	Scale and denoise images using convolutional neural networks
WWW=		https://github.com/DeadSix27/waifu2x-converter-cpp

LICENSE=	BSD2CLAUSE MIT
LICENSE_COMB=	multi
LICENSE_FILE_BSD2CLAUSE=${WRKSRC}/include/picojson_LICENSE.txt
LICENSE_FILE_MIT=	${WRKSRC}/LICENSE

BUILD_DEPENDS=	${LOCALBASE}/include/CL/opencl.h:devel/opencl
LIB_DEPENDS=	libopencv_imgcodecs.so:graphics/opencv

USES=		cmake:testing compiler:c++17-lang localbase:ldflags
USE_GITHUB=	yes
USE_LDCONFIG=	yes
GH_ACCOUNT=	DeadSix27
CMAKE_ON=	INSTALL_MODELS
CMAKE_TESTING_ON=	ENABLE_TESTS
CMAKE_ARGS=	-DOPENCV_PREFIX:PATH="${LOCALBASE}" \
		-DOpenCL_LIBRARY:FILEPATH="${LOCALBASE}/lib/libOpenCL.so"
PLIST_FILES=	bin/${PORTNAME} \
		include/w2xconv.h \
		lib/libw2xc.so
PORTDATA=	*

OPTIONS_DEFINE=	SIMD
OPTIONS_DEFAULT=SIMD

SIMD_CMAKE_OFF=	-DARMOPT:BOOL=false -DPPCOPT:BOOL=false -DX86OPT:BOOL=false

post-patch:
	@${REINPLACE_CMD} -e '/Darwin/,/FLAGS_RELEASE/d' \
		-e 's,".*\(-m[^[:space:]]*\).*","\1",' \
		-e "s/\$${TS}/`${AWK} '/TIMESTAMP/ { print \$$3 }' ${DISTINFO_FILE}`/" \
		-e '/find_program/s/git/&_disabled/' \
		-e '/GIT_BRANCH/s/null/master/' \
		-e '/GIT_COMMIT_HASH/s/000000/${DISTVERSIONSUFFIX:U&:S/^-g//}/' \
		-e '/GIT_TAG/s/v0\.0\.0/${DISTVERSIONFULL:C/-.*//}/' \
		${WRKSRC}/CMakeLists.txt
	@${REINPLACE_CMD} 's,models_rgb,${WRKSRC}/&,' ${WRKSRC}/w32-apps/runtest.c

.include <bsd.port.mk>
