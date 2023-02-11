PORTNAME=	libetonyek
PORTVERSION=	0.1.10
PORTREVISION=	6
PORTEPOCH=	1
CATEGORIES=	graphics
MASTER_SITES=	LODEV/${PORTNAME}
PKGNAMESUFFIX=	01

MAINTAINER=	office@FreeBSD.org
COMMENT=	Library to interpret and import Apple Keynote presentations
WWW=		https://wiki.documentfoundation.org/DLP/Libraries/libetonyek

LICENSE=	MPL20
LICENSE_FILE=	${WRKSRC}/COPYING

BUILD_DEPENDS=	mdds>0:devel/mdds \
		boost-libs>0:devel/boost-libs \
		glm>0:math/glm
LIB_DEPENDS=	librevenge-0.0.so:textproc/librevenge \
		liblangtag.so:devel/liblangtag \
		libxml2.so:textproc/libxml2

USES=		compiler:c++11-lib gmake libtool localbase pkgconfig tar:xz

MDDS_VERSION=	2.0

CONFIGURE_ARGS=	--disable-tests --disable-werror --without-docs --with-mdds=${MDDS_VERSION}

GNU_CONFIGURE=	yes
USE_LDCONFIG=	yes
INSTALL_TARGET=	install-strip

.include <bsd.port.mk>
