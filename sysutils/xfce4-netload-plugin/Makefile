PORTNAME=	xfce4-netload-plugin
PORTVERSION=	1.4.2
CATEGORIES=	sysutils xfce
MASTER_SITES=	XFCE/panel-plugins
DIST_SUBDIR=	xfce4

MAINTAINER=	xfce@FreeBSD.org
COMMENT=	Network Load plugin for Xfce4
WWW=		https://docs.xfce.org/panel-plugins/xfce4-netload-plugin/start

LICENSE=	GPLv2
LICENSE_FILE=	${WRKSRC}/COPYING

LIB_DEPENDS=	libharfbuzz.so:print/harfbuzz

USES=		compiler:c11 gettext-tools gmake gnome libtool pkgconfig \
		tar:bzip2 xfce
USE_GNOME=	cairo gdkpixbuf gtk30
USE_XFCE=	libmenu panel

GNU_CONFIGURE=	yes
INSTALL_TARGET=	install-strip

OPTIONS_DEFINE=	NLS
OPTIONS_SUB=	yes

NLS_USES=		gettext-runtime
NLS_CONFIGURE_ENABLE=	nls

post-patch:
	@${REINPLACE_CMD} -e 's/ || defined \{0,1\}(__FreeBSD_kernel__)//' \
		${WRKSRC}/panel-plugin/net.c \
		${WRKSRC}/panel-plugin/net.h \
		${WRKSRC}/panel-plugin/os.h

.include <bsd.port.mk>
