PORTNAME=	vnote
DISTVERSIONPREFIX=	v
DISTVERSION=	4.2.0
CATEGORIES=	deskutils

MAINTAINER=	danfe@FreeBSD.org
COMMENT=	Pleasant Qt-based note-taking platform
WWW=		https://app.vnote.fun/en_us/

LICENSE=	LGPL3

BUILD_DEPENDS=	pkgconf>0:devel/pkgconf \
		vulkan-headers>0:graphics/vulkan-headers
LIB_DEPENDS=	libhunspell-1.7.so:textproc/hunspell \
		libxkbcommon.so:x11/libxkbcommon

USES=		compiler:c++17-lang cmake cpe desktop-file-utils gl qt:6 xorg
USE_GL=		opengl
USE_QT=		5compat base declarative imageformats pdf positioning \
		sqldriver-sqlite:run svg webchannel webengine
USE_XORG=	ice sm x11 xcb xext
USE_LDCONFIG=	yes

USE_GITHUB=	yes
GH_ACCOUNT=	${PORTNAME}x libgit2:lg2 stdware:qs SineStriker:sc
GH_PROJECT=	QHotkey:qhk qwindowkit:qwk vtextedit:vte vxcore:vc cmark:cm libgit2:lg2 syntax-highlighting:shl sonnet:s qmsetup:qs syscmdline:sc
GH_TAGNAME=	d7b5250:qhk 83d37fc:qwk 6a0f62a:vte ca33f17:vc ecdc4216:cm a418d9d:lg2 25f6871:shl 7320206:s 7854f53:qs 0c9f3de8:sc
GH_SUBDIR=	libs/QHotkey:qhk \
		libs/qwindowkit:qwk \
		libs/qwindowkit/qmsetup:qs \
		libs/qwindowkit/qmsetup/src/syscmdline:sc \
		libs/vtextedit:vte \
		libs/vtextedit/libs/sonnet:s \
		libs/vtextedit/libs/syntax-highlighting:shl \
		libs/vxcore:vc \
		libs/vxcore/third_party/cmark:cm \
		libs/vxcore/third_party/libgit2:lg2 \

CMAKE_ARGS=	-DCMAKE_INSTALL_DATADIR=${DATADIR}

CPE_VENDOR=	${PORTNAME}_project

post-patch:
	@${REINPLACE_CMD} -E '/add_subdirectory\((demo|tests)\)/d' \
		${WRKSRC}/libs/vtextedit/CMakeLists.txt \
		${WRKSRC}/CMakeLists.txt
	@${GREP} -Rl Q_OS_LINUX ${WRKSRC}/libs ${WRKSRC}/src | ${GREP} -E '\.[ch](pp)?$$' | ${XARGS} \
		${REINPLACE_CMD} -e 's,Q_OS_LINUX,Q_OS_UNIX,'
	@${REINPLACE_CMD} -e 's,%%DATADIR%%,${DATADIR},' \
		${WRKSRC}/src/core/configmgr.cpp ${WRKSRC}/src/core/configmgr2.cpp
	@${RM} -rf ${WRKSRC}/libs/vtextedit/libs/cmark
	@${LN} -sf ${WRKSRC}/libs/vxcore/third_party/cmark ${WRKSRC}/libs/vtextedit/libs/cmark

.include <bsd.port.mk>
