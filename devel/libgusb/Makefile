PORTNAME=	libgusb
PORTVERSION=	0.4.5
CATEGORIES=	devel

MAINTAINER=	gnome@FreeBSD.org
COMMENT=	GUsb is a GObject wrapper for libusb1
WWW=		https://github.com/hughsie/libgusb

LICENSE=	LGPL21
LICENSE_FILE=	${WRKSRC}/COPYING

BUILD_DEPENDS=	gtk-doc>0:textproc/gtk-doc
LIB_DEPENDS=	libjson-glib-1.0.so:devel/json-glib

USES=		gnome meson pkgconfig python:build vala:build

USE_GITHUB=	yes
GH_ACCOUNT=	hughsie

USE_GNOME=	glib20 introspection:build

OPTIONS_DEFINE=		DOCS VAPI
OPTIONS_DEFAULT=	DOCS VAPI
OPTIONS_SUB=		yes

DOCS_BUILD_DEPENDS=	${PYTHON_PKGNAMEPREFIX}gi-docgen>=0:textproc/py-gi-docgen@${PY_FLAVOR}

.include <bsd.port.options.mk>

MESON_ARGS+=	-Dumockdev=disabled
.if !${PORT_OPTIONS:MDOCS}
MESON_ARGS+=	-Ddocs=false
.endif
.if !${PORT_OPTIONS:MVAPI}
MESON_ARGS+=	-Dvapi=false
.endif

.include <bsd.port.mk>
