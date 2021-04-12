# Created by: Alexey Dokuchaev <danfe@FreeBSD.org>
<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	storj
PORTVERSION=	1.25.2
=======

PORTNAME=	storj
PORTVERSION=	1.26.3
>>>>>>> upstream/main
DISTVERSIONPREFIX=	v
CATEGORIES=	net

MAINTAINER=	danfe@FreeBSD.org
COMMENT=	Decentralized cloud object storage

LICENSE=	AGPLv3

USES=		go:modules
USE_GITHUB=	yes

.include "${.CURDIR}/Makefile.deps"

<<<<<<< HEAD
STORJ_CMDS=	certificates identity inspector metric-receiver multinode \
		satellite segment-reaper storagenode storagenode-updater \
=======
STORJ_CMDS=	certificates identity inspector metainfo-loop-benchmark \
		metainfo-migration metainfo-verifier metric-receiver \
		multinode satellite storagenode storagenode-updater \
>>>>>>> upstream/main
		storj-admin storj-sim uplink versioncontrol

GO_TARGET=	${STORJ_CMDS:S,^,./cmd/,}

post-stage:
# Avoid conflict with `graphics/mesa-demos'
	@${MV} ${STAGEDIR}${PREFIX}/bin/identity \
		${STAGEDIR}${PREFIX}/bin/${PORTNAME}-identity

.include <bsd.port.mk>
