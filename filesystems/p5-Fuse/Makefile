PORTNAME=	Fuse
PORTVERSION=	0.16.1
PORTREVISION=	2
CATEGORIES=	filesystems sysutils perl5
MASTER_SITES=	CPAN
MASTER_SITE_SUBDIR=	CPAN:DPATES
PKGNAMEPREFIX=	p5-

MAINTAINER=	perl@FreeBSD.org
COMMENT=	Write filesystems in Perl using FUSE
WWW=		https://metacpan.org/release/Fuse

LICENSE=	LGPL21

USES=		perl5 fuse pkgconfig
USE_PERL5=	configure

PORTSCOUT=	limit:^[0-9\.]*$$

post-install:
	${STRIP_CMD} ${STAGEDIR}${PREFIX}/${SITE_ARCH_REL}/auto/Fuse/Fuse.so

.include <bsd.port.mk>
