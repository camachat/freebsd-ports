PORTNAME=	voro++
DISTVERSION=	0.4.6
PORTREVISION=	2
CATEGORIES=	science
MASTER_SITES=	http://math.lbl.gov/voro++/download/dir/

MAINTAINER=	jwb@FreeBSD.org
COMMENT=	Three-dimensional computations of the Voronoi tessellation
WWW=		http://math.lbl.gov/voro++/

LICENSE=	BSD3CLAUSE
LICENSE_FILE=	${WRKSRC}/LICENSE

USES=		gmake
USE_LDCONFIG=	yes

CFLAGS+=	-fPIC

post-install:
	@${STRIP_CMD} ${STAGEDIR}${PREFIX}/bin/voro++

.include <bsd.port.mk>
