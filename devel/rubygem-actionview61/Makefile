PORTNAME=	actionview
PORTVERSION=	6.1.7.10
CATEGORIES=	devel rubygems
MASTER_SITES=	RG
PKGNAMESUFFIX=	61

MAINTAINER=	sunpoet@FreeBSD.org
COMMENT=	Rendering framework putting the V in MVC (part of Rails)
WWW=		https://github.com/rails/rails/tree/main/actionview \
		https://rubyonrails.org/

LICENSE=	MIT
LICENSE_FILE=	${WRKSRC}/MIT-LICENSE

RUN_DEPENDS=	rubygem-activesupport61>=${PORTVERSION}<${PORTVERSION}_99:devel/rubygem-activesupport61 \
		rubygem-builder>=3.1<4:devel/rubygem-builder \
		rubygem-erubi>=1.4<2:www/rubygem-erubi \
		rubygem-rails-dom-testing-rails61>=2.0<3:textproc/rubygem-rails-dom-testing-rails61 \
		rubygem-rails-html-sanitizer>=1.2.0<2:textproc/rubygem-rails-html-sanitizer

USES=		cpe gem

NO_ARCH=	yes

CPE_VENDOR=	rubyonrails

PORTSCOUT=	limit:^6\.1\.

.include <bsd.port.mk>
