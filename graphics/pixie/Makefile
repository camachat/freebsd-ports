# Created by: Igor Pokrovsky <tiamat@comset.net>
<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	pixie
PORTVERSION=	2.2.6
PORTREVISION=	29
=======

PORTNAME=	pixie
PORTVERSION=	2.2.6
PORTREVISION=	30
>>>>>>> upstream/main
CATEGORIES=	graphics
MASTER_SITES=	SF/${PORTNAME}/${PORTNAME}/Pixie%20${PORTVERSION}
DISTNAME=	Pixie-src-${PORTVERSION}

MAINTAINER=	ports@FreeBSD.org
COMMENT=	Photorealistic renderer with Pixar's RenderMan-like interface

LICENSE=	LGPL21+
LICENSE_FILE=	${WRKSRC}/COPYING

BROKEN_aarch64=		fails to compile: ri/atomic.h:149:9: use of undeclared identifier CRenderer
BROKEN_armv6=		fails to compile: ri/atomic.h:149:9: use of undeclared identifier CRenderer
BROKEN_armv7=		fails to compile: ri/atomic.h:149:9: use of undeclared identifier CRenderer
BROKEN_riscv64=		fails to compile: ri/atomic.h:149:9: use of undeclared identifier CRenderer
BROKEN_FreeBSD_11_powerpc64=	fails to compile on powerpc64 elfv1: ../common/mathSpec.h:548:37: error: 'max' was not declared in this scope
BROKEN_FreeBSD_12_powerpc64=	fails to compile on powerpc64 elfv1: ../common/mathSpec.h:548:37: error: 'max' was not declared in this scope

<<<<<<< HEAD
LIB_DEPENDS=	libIlmImf.so:graphics/openexr \
		libtiff.so:graphics/tiff \
		libfltk_gl.so:x11-toolkits/fltk

USES=		bison compiler:c++0x gl libtool pathfix tar:tgz
=======
LIB_DEPENDS=	libOpenEXR.so:graphics/openexr \
		libImath.so:math/Imath \
		libtiff.so:graphics/tiff \
		libfltk_gl.so:x11-toolkits/fltk

USES=		autoreconf bison compiler:c++0x gl libtool pathfix tar:tgz
>>>>>>> upstream/main
USE_GL=		glu
GNU_CONFIGURE=	yes
CONFIGURE_ARGS=	--includedir=${PREFIX}/include/pixie \
		--libdir=${PREFIX}/lib/pixie \
		--with-docdir=${DOCSDIR} \
		--with-shaderdir=${DATADIR}/shaders \
		--with-ribdir=${DATADIR}/ribs \
		--with-texturedir=${DATADIR}/textures \
		--with-displaysdir=${PREFIX}/lib/pixie/displays \
		--with-modulesdir=${PREFIX}/lib/pixie/modules \
		--with-openexr-prefix=${LOCALBASE} \
		--enable-openexr-threads
WRKSRC=		${WRKDIR}/Pixie
INSTALL_TARGET=	install-strip
USE_LDCONFIG=	${PREFIX}/lib/${PORTNAME}

PORTDOCS=	*

CFLAGS+=	-fPIC
<<<<<<< HEAD
CPPFLAGS+=	-I${LOCALBASE}/include
=======
CPPFLAGS+=	-I${LOCALBASE}/include -I${LOCALBASE}/include/Imath
>>>>>>> upstream/main
LDFLAGS+=	-L${LOCALBASE}/lib

OPTIONS_DEFINE=	DOCS

post-patch:
	@${REINPLACE_CMD} -e '/test/s|==|=|g ; \
		 s|-lpthread|-pthread|g' ${WRKSRC}/configure
<<<<<<< HEAD
	@${REINPLACE_CMD} -e '/^SUBDIRS/s|doc||g ; \
		 /^install-data-am:/s|install-nobase_docDATA||g' \
		 ${WRKSRC}/Makefile.in
	@${REINPLACE_CMD} -e '/^includedir/s|=.*$$|= @includedir@|g' \
		${WRKSRC}/src/ri/Makefile.in \
		${WRKSRC}/src/sdr/Makefile.in
=======
	@${REINPLACE_CMD} -e '/^includedir/s|=.*$$|= @includedir@|g' \
		${WRKSRC}/src/ri/Makefile.am \
		${WRKSRC}/src/sdr/Makefile.am
>>>>>>> upstream/main

post-install-DOCS-on:
	@${MKDIR} ${STAGEDIR}${DOCSDIR}
	(cd ${WRKSRC} && ${INSTALL_DATA} AUTHORS ChangeLog DEVNOTES NEWS \
		README ${STAGEDIR}${DOCSDIR})
	(cd ${WRKSRC}/doc && ${COPYTREE_SHARE} . ${STAGEDIR}${DOCSDIR} \
		'! -name "*Makefile*"')

.include <bsd.port.mk>
