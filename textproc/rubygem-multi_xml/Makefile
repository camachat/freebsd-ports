PORTNAME=	multi_xml
PORTVERSION=	0.7.2
CATEGORIES=	textproc rubygems
MASTER_SITES=	RG

MAINTAINER=	sunpoet@FreeBSD.org
COMMENT=	Ruby library provide swappable XML backends
WWW=		https://github.com/sferik/multi_xml

LICENSE=	MIT
LICENSE_FILE=	${WRKSRC}/LICENSE.md

RUN_DEPENDS=	rubygem-bigdecimal>=3.1<4:math/rubygem-bigdecimal

USES=		cpe gem

NO_ARCH=	yes

CPE_VENDOR=	erik_michaels-ober

OPTIONS_DEFINE=	NOKOGIRI
NOKOGIRI_DESC=	Use Nokogiri for parsing

NOKOGIRI_RUN_DEPENDS=	rubygem-nokogiri>=0:textproc/rubygem-nokogiri

.include <bsd.port.mk>
