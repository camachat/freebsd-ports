# Created by: Muhammad Moinur Rahman <bofh@FreeBSD.org>
<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	mroonga
PORTVERSION=	11.00
=======

PORTNAME=	mroonga
PORTVERSION=	11.01
>>>>>>> upstream/main
CATEGORIES=	databases textproc
MASTER_SITES=	http://packages.groonga.org/source/mroonga/

MAINTAINER=	bofh@FreeBSD.org
COMMENT=	Open-source fulltext search engine for MySQL

LICENSE=	LGPL21

IGNORE_WITH_MYSQL=	101m 102m 103m 57p 57w

BUILD_DEPENDS=	${NONEXISTENT}:${_MYSQL_SERVER}:configure
LIB_DEPENDS=	libgroonga.so:textproc/groonga

<<<<<<< HEAD
USES=		compiler:c11 libtool mysql pkgconfig
=======
USES=		compiler:c11 libtool mysql pkgconfig shebangfix

SHEBANG_FILES=	data/rpm/post.sh \
				data/rpm/preun.sh
>>>>>>> upstream/main

GNU_CONFIGURE=	yes
CONFIGURE_ARGS=	--with-mysql-source="$$(cd ${PORTSDIR}/${_MYSQL_SERVER}; ${MAKE} -V WRKSRC)" \
		--with-mysql-build="$$(cd ${PORTSDIR}/${_MYSQL_SERVER}; ${MAKE} -V BUILD_WRKSRC)" \
		--with-mysql-config=${PREFIX}/bin/mysql_config

SUB_FILES=	pkg-message

OPTIONS_DEFINE=		DOCS

post-patch:
# Avoid conflict with C++20 <version> by adding .txt suffix
	@${MV} ${WRKSRC}/version ${WRKSRC}/version.txt
	@${REINPLACE_CMD} -i .c++20 's,srcdir)/version[[:>:]],&.txt,' \
		${WRKSRC}/Makefile.in

post-install:
	${STRIP_CMD} ${STAGEDIR}${PREFIX}/lib/mysql/plugin/ha_mroonga.so.0.0.0

.include <bsd.port.mk>
