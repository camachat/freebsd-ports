# Created by: Yuri Victorovich <yuri@rawbw.com>
<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	caps-lv2
PORTVERSION=	g20190321
=======

PORTNAME=	caps-lv2
PORTVERSION=	g20200529
>>>>>>> upstream/main
CATEGORIES=	audio

MAINTAINER=	yuri@FreeBSD.org
COMMENT=	LV2 version of the C* Audio Plugin Suite

LICENSE=	GPLv3
LICENSE_FILE=	${WRKSRC}/COPYING

BUILD_DEPENDS=	lv2>0:audio/lv2

USES=		gmake localbase
USE_GITHUB=	yes
GH_ACCOUNT=	moddevices
<<<<<<< HEAD
GH_TAGNAME=	250844a

post-patch:
	@${REINPLACE_CMD} -e "s|__APPLE__|__FreeBSD__|" \
		${WRKSRC}/dsp/v4f_IIR2.h
=======
GH_TAGNAME=	5d52a0c
>>>>>>> upstream/main

.include <bsd.port.mk>
