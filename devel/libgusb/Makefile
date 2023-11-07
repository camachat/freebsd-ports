PORTNAME=	libgusb
PORTVERSION=	0.4.7
CATEGORIES=	devel

MAINTAINER=	gnome@FreeBSD.org
COMMENT=	GUsb is a GObject wrapper for libusb1
WWW=		https://github.com/hughsie/libgusb

LICENSE=	LGPL21
LICENSE_FILE=	${WRKSRC}/COPYING

BUILD_DEPENDS=	${PY_SETUPTOOLS}
LIB_DEPENDS=	libjson-glib-1.0.so:devel/json-glib

USES=		gnome meson pkgconfig python:build

USE_GITHUB=	yes
GH_ACCOUNT=	hughsie

USE_GNOME=	glib20 introspection:build

OPTIONS_DEFINE=		DOCS VAPI
OPTIONS_DEFAULT=	DOCS VAPI
OPTIONS_SUB=		yes

MESON_ARGS+=		-Dumockdev=disabled

DOCS_MESON_TRUE=	docs
DOCS_BUILD_DEPENDS=	gtk-doc>0:textproc/gtk-doc \
			${PYTHON_PKGNAMEPREFIX}gi-docgen>=0:textproc/py-gi-docgen@${PY_FLAVOR}

VAPI_MESON_TRUE=	vapi
VAPI_USES=		vala:build

.include <bsd.port.mk>
