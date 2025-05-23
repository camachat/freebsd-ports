PORTNAME=	parole
PORTVERSION=	4.18.2
PORTREVISION=	1
CATEGORIES=	multimedia xfce
MASTER_SITES=	XFCE/apps
DIST_SUBDIR=	xfce4

MAINTAINER=	xfce@FreeBSD.org
COMMENT=	Lightweight media player for Xfce4 based on GStreamer
WWW=		https://docs.xfce.org/apps/parole/start

LICENSE=	GPLv2
LICENSE_FILE=	${WRKSRC}/COPYING

LIB_DEPENDS=	libdbus-1.so:devel/dbus \
		libdbus-glib-1.so:devel/dbus-glib \
		libnotify.so:devel/libnotify \
		libtag_c.so:audio/taglib \
		libharfbuzz.so:print/harfbuzz
RUN_DEPENDS=	xdg-screensaver:devel/xdg-utils

USES=		compiler:c++11-lang desktop-file-utils gettext-tools \
		gmake gnome gstreamer libtool pkgconfig tar:bzip2 xfce xorg
USE_GNOME=	cairo gdkpixbuf gtk30 glib20
USE_GSTREAMER=	core faad flac wavpack x
USE_XFCE=	libmenu xfconf
USE_XORG=	ice sm x11

GNU_CONFIGURE=	yes
CONFIGURE_ARGS=	--enable-notify-plugin \
		--enable-tray-plugin \
		--enable-taglib \
		--enable-mpris2-plugin \
		--without-html-dir
INSTALL_TARGET=	install-strip

OPTIONS_DEFINE=		NLS
OPTIONS_DEFAULT=	HTTP
OPTIONS_GROUP=		GST_PLUG
OPTIONS_GROUP_GST_PLUG=	CDDA HTTP
OPTIONS_SUB=		yes

GST_PLUG_DESC=		Additional GStreamer plugins
HTTP_DESC=		HTTP streaming (libsoup)

CDDA_USE=		gstreamer=cdparanoia

HTTP_USE=		gstreamer=soup

NLS_USES=		gettext-runtime
NLS_CONFIGURE_ENABLE=	nls

post-patch:
# Fix .rc files path
	@${REINPLACE_CMD} -e 's|xfce4/src/misc|${PORTNAME}|g' \
		${WRKSRC}/src/common/parole-rc-utils.h

.include <bsd.port.mk>
