# Created by: Sean McGovern <sean@sfarc.net>
<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	faad2
PORTVERSION=	2.8.8
PORTEPOCH=	1
CATEGORIES=	audio
MASTER_SITES=	SF/faac/${PORTNAME}-src/${PORTNAME}-${PORTVERSION:R}.0
=======

PORTNAME=	faad2
PORTVERSION=	2.10.0
PORTEPOCH=	1
CATEGORIES=	audio
>>>>>>> upstream/main

MAINTAINER=	multimedia@FreeBSD.org
COMMENT=	MPEG-2 and MPEG-4 AAC audio decoder

LICENSE=	GPLv2+
LICENSE_FILE=	${WRKSRC}/COPYING

USES=		autoreconf gmake libtool localbase:ldflags
INSTALL_TARGET=	install-strip

GNU_CONFIGURE=	yes
LDFLAGS+=	-lpthread
USE_LDCONFIG=	yes

<<<<<<< HEAD
=======
USE_GITHUB=	yes
GH_ACCOUNT=	knik0
GH_TAGNAME=	c626366

>>>>>>> upstream/main
post-extract:
	@${LN} -sf faad.man ${WRKSRC}/frontend/faad.1

post-patch:
	@${REINPLACE_CMD} -e 's|faad.man|faad.1|g' \
		${WRKSRC}/frontend/Makefile.am
	@${REINPLACE_CMD} -e 's|^AM_CFLAGS|AM_CPPFLAGS|g ; \
		 s|-iquote |-I|g' ${WRKSRC}/libfaad/Makefile.am
<<<<<<< HEAD
	@${REINPLACE_CMD} -e 's|= -DUSE_|= -fPIC -DPIC -DUSE_|g' \
		${WRKSRC}/common/mp4ff/Makefile.am
	@${ECHO_CMD} "SUBDIRS = libfaad common frontend" > \
		${WRKSRC}/Makefile.am
	@${REINPLACE_CMD} -E -e 's|(FAAD2_VERSION +)"unknown"|\1"${PORTVERSION:R}"|' \
		${WRKSRC}/include/neaacdec.h

post-configure:
	@${REINPLACE_CMD} "s/444/644/g" ${WRKSRC}/common/mp4ff/Makefile

=======
	@${REINPLACE_CMD} -E -e 's|(FAAD2_VERSION +)"unknown"|\1"${PORTVERSION:R}"|' \
		${WRKSRC}/include/neaacdec.h

>>>>>>> upstream/main
.include <bsd.port.mk>
