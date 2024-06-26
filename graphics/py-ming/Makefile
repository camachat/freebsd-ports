PORTNAME=	ming
DISTVERSIONPREFIX=	ming-
DISTVERSION=	0_4_8
CATEGORIES=	graphics python
PKGNAMEPREFIX=	${PYTHON_PKGNAMEPREFIX}

MAINTAINER=	sunpoet@FreeBSD.org
COMMENT=	Python module for Ming which allows you to create Flash 4/5 movies
WWW=		https://github.com/libming/libming

LICENSE=	GPLv2 LGPL21
LICENSE_COMB=	multi
LICENSE_FILE_GPLv2=	${WRKSRC}/LICENSE_GPL2
LICENSE_FILE_LGPL21=	${WRKSRC}/LICENSE

DEPRECATED=	Depends on deprecated graphics/ming
EXPIRATION_DATE=2024-06-30

LIB_DEPENDS=	libfreetype.so:print/freetype2 \
		libgif.so:graphics/giflib \
		libming.so:graphics/ming \
		libpng.so:graphics/png

USES=		autoreconf cpe libtool localbase:ldflags pkgconfig python:3.8-3.9
USE_PYTHON=	autoplist concurrent distutils

CONFIGURE_ARGS=	--enable-python
CONFIGURE_ENV=	PYTHON_LDFLAGS="-L${PYTHON_LIBDIR} -l${PYTHON_VERSION}${PYTHON_ABIVER}"
GNU_CONFIGURE=	yes
CPE_VENDOR=	libming

BUILD_WRKSRC=	${WRKSRC}/py_ext
INSTALL_WRKSRC=	${WRKSRC}/py_ext

USE_GITHUB=	yes
GH_ACCOUNT=	libming
GH_PROJECT=	libming

PORTSCOUT=	limit:^ming-

post-patch:
	@${REINPLACE_CMD} -e 's|/usr/local/|${LOCALBASE}/|g' ${WRKSRC}/py_ext/setup.py.in
	@${LN} -fs ${LOCALBASE}/include/ming ${WRKSRC}/src

post-install:
	${FIND} ${STAGEDIR}${PYTHON_SITELIBDIR} -name '*.so' -exec ${STRIP_CMD} {} +

.include <bsd.port.mk>
