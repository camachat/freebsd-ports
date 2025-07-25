PORTNAME=	flickcurl
PORTVERSION=	1.26
PORTREVISION=	5
CATEGORIES=	www
MASTER_SITES=	http://download.dajobe.org/flickcurl/

MAINTAINER=	ports@FreeBSD.org
COMMENT=	C library for the Flickr API
WWW=		https://librdf.org/flickcurl/

LICENSE=	APACHE20 GPLv2 LGPL21
LICENSE_COMB=	dual

LIB_DEPENDS=	libcurl.so:ftp/curl

OPTIONS_DEFINE=	RAPTOR DOCS
OPTIONS_RADIO=	WEBSERVICE
OPTIONS_RADIO_WEBSERVICE=	OFFLINE CAPTURE
OPTIONS_DEFAULT=RAPTOR
RAPTOR_DESC=	Use Raptor to serialize to triples
OFFLINE_DESC=	Offline use of captured XML web service responses
CAPTURE_DESC=	Capturing web service responses

USES=		gnome libtool pathfix pkgconfig
USE_GNOME=	libxml2
GNU_CONFIGURE=	yes
GNU_CONFIGURE_MANPREFIX=${PREFIX}/share
CONFIGURE_ARGS=	--with-html-dir=${PREFIX}/share/doc
USE_LDCONFIG=	yes
INSTALL_TARGET=	install-strip

OFFLINE_CONFIGURE_ON=		--enable-offline
OFFLINE_CONFIGURE_OFF=		offline=no
CAPTURE_CONFIGURE_ON=		--enable-capture
CAPTURE_CONFIGURE_OFF=		capture=no
RAPTOR_LIB_DEPENDS=		libraptor2.so:textproc/raptor2
RAPTOR_CONFIGURE_ON=		--with-raptor=2
RAPTOR_CONFIGURE_OFF=		--with-raptor=no

.include <bsd.port.mk>
