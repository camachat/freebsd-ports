PORTNAME=	wlr-which-key
DISTVERSIONPREFIX=	v
DISTVERSION=	1.2.0
CATEGORIES=	x11 wayland

MAINTAINER=	ports@FreeBSD.org
COMMENT=	Keymap manager for wlroots-based compositors
WWW=		https://github.com/MaxVerevkin/wlr-which-key

LICENSE=	GPLv3
LICENSE_FILE=	${WRKSRC}/LICENSE

LIB_DEPENDS=	libxkbcommon.so:x11/libxkbcommon

USES=		cargo gnome
USE_GITHUB=	yes
USE_GNOME=	cairo pango
GH_ACCOUNT=	MaxVerevkin
PLIST_FILES=	bin/${PORTNAME}

.include <bsd.port.mk>
