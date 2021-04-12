<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	libxkbcommon
PORTVERSION=	1.1.0
=======
PORTNAME=	libxkbcommon
PORTVERSION=	1.2.1
>>>>>>> upstream/main
CATEGORIES=	x11
MASTER_SITES=	https://xkbcommon.org/download/
DIST_SUBDIR=	xorg/lib

MAINTAINER=	x11@FreeBSD.org
COMMENT=	Keymap handling library for toolkits and window systems

LICENSE=	MIT
LICENSE_FILE=	${WRKSRC}/LICENSE

RUN_DEPENDS=	xkeyboard-config>0:x11/xkeyboard-config

<<<<<<< HEAD
USES=		bison gnome meson pkgconfig tar:xz
USE_GNOME=	libxml2
MESON_ARGS=	-Denable-docs=false \
		-Denable-wayland=false \
		-Dx-locale-root="${PREFIX}/lib/X11/locale"

OPTIONS_DEFINE=	X11
OPTIONS_DEFAULT=X11
OPTIONS_SUB=	yes

=======
USES=		bison compiler:c11 gnome meson pkgconfig tar:xz
USE_GNOME=	libxml2
USE_LDCONFIG=	yes
MESON_ARGS=	-Denable-docs=false \
		-Dx-locale-root="${PREFIX}/lib/X11/locale" \
		${MESON_ARGS_${OPSYS}_${OSREL:R}}
MESON_ARGS_FreeBSD_11=	-Ddefault-rules=xorg

OPTIONS_DEFINE=	EVDEV WAYLAND X11
OPTIONS_DEFAULT=EVDEV WAYLAND X11
OPTIONS_SUB=	yes

EVDEV_DESC=		Interactive debugger for XKB keymaps for evdev
EVDEV_BUILD_DEPENDS=	evdev-proto>0:devel/evdev-proto
EVDEV_USES=		localbase

WAYLAND_BUILD_DEPENDS=	wayland-protocols>=1.12:graphics/wayland-protocols
WAYLAND_LIB_DEPENDS=	libwayland-client.so:graphics/wayland
WAYLAND_MESON_TRUE=	enable-wayland

>>>>>>> upstream/main
X11_USES=	xorg
X11_USE=	XORG=xcb
X11_MESON_TRUE=	enable-x11

<<<<<<< HEAD
.include <bsd.port.pre.mk>

.if ${OSVERSION} < 1200000
MESON_ARGS+=	-Ddefault-rules=xorg
.endif

.include <bsd.port.post.mk>
=======
.include <bsd.port.mk>
>>>>>>> upstream/main
