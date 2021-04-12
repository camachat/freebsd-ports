# Created by: Andrew Pantyukhin <infofarmer@FreeBSD.org>
<<<<<<< HEAD
# $FreeBSD$

PORTNAME?=	Net-SinFP
PORTVERSION=	2.09
.ifdef DBDATE
PORTREVISION=	${DBDATE}1
.else
PORTREVISION=	1
.endif
CATEGORIES=	security perl5
MASTER_SITES=	CPAN
.ifdef DBDATE
MASTER_SITES+=	http://www.gomor.org/files/:DB
.endif
MASTER_SITE_SUBDIR=	CPAN:GOMOR
DISTNAME=	Net-SinFP-${PORTVERSION}
.ifdef DBDATE
DISTFILES=	${DISTNAME}${EXTRACT_SUFX} \
		${PORTNAME}-${DBDATE}.db:DB
EXTRACT_ONLY=	${DISTNAME}${EXTRACT_SUFX}
.endif
=======

PORTNAME=	Net-SinFP
PORTVERSION=	2.10
CATEGORIES=	security perl5
MASTER_SITES=	CPAN
PKGNAMEPREFIX=	p5-
>>>>>>> upstream/main

MAINTAINER=	sbz@FreeBSD.org
COMMENT=	Full operating system stack fingerprinting suite

LICENSE=	ART10
LICENSE_FILE=	${WRKSRC}/LICENSE.Artistic

<<<<<<< HEAD
.if ${PORTNAME} == "Net-SinFP"
PKGNAMEPREFIX=	p5-

BUILD_DEPENDS=	${RUN_DEPENDS}
RUN_DEPENDS=	p5-Class-Gomor>=1.00:devel/p5-Class-Gomor \
		p5-DBIx-SQLite-Simple>=0.33:databases/p5-DBIx-SQLite-Simple \
		p5-Net-Packet>=3.00:net/p5-Net-Packet \
		p5-Net-Packet-Target>=1.00:net/p5-Net-Packet-Target

USES=		perl5
USE_PERL5=	configure
INSTALL_TARGET=	pure_install

post-patch:
	@${REINPLACE_CMD} -e 's/.*EXE_FILES.*//' ${WRKSRC}/${CONFIGURE_SCRIPT}
.endif
=======
BUILD_DEPENDS=	${RUN_DEPENDS}
RUN_DEPENDS=	p5-Class-Gomor>=1.00:devel/p5-Class-Gomor \
		p5-DBIx-SQLite-Simple>=0.33:databases/p5-DBIx-SQLite-Simple \
		p5-Net-Packet-Target>=1.00:net/p5-Net-Packet-Target \
		p5-Net-Packet>=3.20:net/p5-Net-Packet

USES=		perl5
USE_PERL5=	configure

INSTALL_TARGET=	pure_install

NO_ARCH=	yes

post-patch:
	@${REINPLACE_CMD} -e 's/.*EXE_FILES.*//' ${WRKSRC}/${CONFIGURE_SCRIPT}
>>>>>>> upstream/main

.include <bsd.port.mk>
