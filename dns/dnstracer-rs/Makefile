PORTNAME=	dnstracer
DISTVERSION=	1.1.5
PORTREVISION=	1
CATEGORIES=	dns
MASTER_SITES=	CRATESIO
PKGNAMESUFFIX=	-rs
DISTFILES=	${CARGO_DIST_SUBDIR}/${DISTNAME}${CARGO_CRATE_EXT}

MAINTAINER=	mat@FreeBSD.org
COMMENT=	Trace a chain of DNS servers to the source
WWW=		https://gitlab.com/mat813/dnstracer-rs

LICENSE=	MIT

USES=		cargo

PLIST_FILES=	bin/dnstracer

# make cargo-crates > Makefile.crates

.include <bsd.port.mk>
