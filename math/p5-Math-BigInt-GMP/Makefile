PORTNAME=	Math-BigInt-GMP
PORTVERSION=	1.7003
CATEGORIES=	math perl5
MASTER_SITES=	CPAN
PKGNAMEPREFIX=	p5-

MAINTAINER=	perl@FreeBSD.org
COMMENT=	Use GMP library for Math::BigInt routines
WWW=		https://metacpan.org/release/Math-BigInt-GMP

LICENSE=	ART10 GPLv1+
LICENSE_COMB=	dual

BUILD_DEPENDS=	${RUN_DEPENDS}
LIB_DEPENDS=	libgmp.so:math/gmp
RUN_DEPENDS=	p5-Math-BigInt>=2.005001:math/p5-Math-BigInt

USES=		localbase perl5
USE_PERL5=	configure

post-install:
	${STRIP_CMD} ${STAGEDIR}${PREFIX}/${SITE_ARCH_REL}/auto/Math/BigInt/GMP/GMP.so

.include <bsd.port.mk>
