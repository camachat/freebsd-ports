PORTNAME=	xdx
DISTVERSION=	2.91
PORTREVISION=	3
CATEGORIES=	comms hamradio
MASTER_SITES=	SF/xdxclusterclient/${PORTVERSION}/

MAINTAINER=	hamradio@FreeBSD.org
COMMENT=	Amateur Radio DX cluster monitor
WWW=		https://sourceforge.net/projects/xdxclusterclient/

LICENSE=	GPLv2

LIB_DEPENDS=	libfontconfig.so:x11-fonts/fontconfig \
		libfreetype.so:print/freetype2

USES=		gettext gmake gnome pkgconfig
USE_GNOME=	cairo gdkpixbuf gtk30

GNU_CONFIGURE=	yes
GNU_CONFIGURE_MANPREFIX=${PREFIX}/share

.include <bsd.port.mk>
