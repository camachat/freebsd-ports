<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	geonkick
DISTVERSIONPREFIX=	v
DISTVERSION=	2.7.2
CATEGORIES=	audio
PKGNAMESUFFIX=	-lv2

=======
PORTNAME=	geonkick
DISTVERSIONPREFIX=	v
DISTVERSION=	2.8.0
CATEGORIES=	audio
PKGNAMESUFFIX=	-lv2

PATCH_SITES=	https://github.com/${GH_ACCOUNT}/${GH_PROJECT}/commit/
PATCHFILES+=	9dd43995ec37853893db14bce7f420a03ff4194b.patch:-p1 # Support redkite on FreeBSD: https://github.com/iurie-sw/geonkick/pull/160

>>>>>>> upstream/main
MAINTAINER=	yuri@FreeBSD.org
COMMENT=	Software percussion synthesizer

LICENSE=	GPLv3
LICENSE_FILE=	${WRKSRC}/LICENSE

BUILD_DEPENDS=	lv2>0:audio/lv2 \
		gmake:devel/gmake \
<<<<<<< HEAD
		rapidjson>0:devel/rapidjson \
		${LOCALBASE}/lib/libredkite.a:x11-toolkits/redkite
=======
		rapidjson>0:devel/rapidjson
>>>>>>> upstream/main
LIB_DEPENDS=	libjack.so:audio/jack \
		libsndfile.so:audio/libsndfile

USES=		compiler:c++17-lang cmake desktop-file-utils gnome localbase:ldflags \
		pkgconfig shared-mime-info xorg
USE_GITHUB=	yes
GH_ACCOUNT=	iurie-sw
USE_GNOME=	cairo
USE_XORG=	x11

INSTALLS_ICONS=	yes

.include <bsd.port.mk>
