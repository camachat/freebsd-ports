PORTNAME=	pg_query
PORTVERSION=	6.1.0
CATEGORIES=	databases rubygems
MASTER_SITES=	RG

MAINTAINER=	mfechner@FreeBSD.org
COMMENT=	Parses SQL queries using a copy of the PostgreSQL server query parser
WWW=		https://github.com/lfittl/pg_query

LICENSE=	BSD3CLAUSE
LICENSE_FILE=	${WRKSRC}/LICENSE

BUILD_DEPENDS=	rubygem-pathname>0:devel/rubygem-pathname
RUN_DEPENDS=	rubygem-google-protobuf>=3.25.3:devel/rubygem-google-protobuf

USES=		gem

post-patch:
#	${RM} ${WRKSRC}/ext/pg_query/pg_query_ruby_freebsd.sym

.include <bsd.port.mk>
