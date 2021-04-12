<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	json-yaml
DISTVERSION=	1.1.2
=======
PORTNAME=	json-yaml
DISTVERSION=	1.2.1
>>>>>>> upstream/main
CATEGORIES=	textproc

MAINTAINER=	ik@sjmulder.nl
COMMENT=	Convert JSON to YAML

LICENSE=	BSD2CLAUSE
LICENSE_FILE=	${WRKSRC}/LICENSE.md

LIB_DEPENDS=	libyajl.so:devel/yajl \
		libyaml.so:textproc/libyaml

<<<<<<< HEAD
MAKE_ARGS+=	MANPREFIX=${PREFIX}/man

USE_GITHUB=	yes
GH_ACCOUNT=	sjmulder

PLIST_FILES=	bin/json-yaml \
		man/man1/json-yaml.1.gz
=======
USE_GITHUB=	yes
GH_ACCOUNT=	sjmulder

MAKE_ARGS+=	MANPREFIX=${PREFIX}/man CFLAGS=-I${PREFIX}/include\ -L${PREFIX}/lib

PLIST_FILES=	bin/json-yaml \
		bin/yaml-json \
		man/man1/json-yaml.1.gz \
		man/man1/yaml-json.1.gz
>>>>>>> upstream/main

.include <bsd.port.mk>
