PORTNAME=	magick
DISTVERSION=	2.8.5
PORTREVISION=	1
CATEGORIES=	graphics
DISTNAME=	${PORTNAME}_${DISTVERSION}

MAINTAINER=	uzsolt@FreeBSD.org
COMMENT=	Advanced Graphics and Image-Processing in R
WWW=		https://cran.r-project.org/web/packages/magick/index.html

LICENSE=	MIT

CRAN_DEPENDS=	R-cran-curl>0:ftp/R-cran-curl \
		R-cran-magrittr>0:devel/R-cran-magrittr \
		R-cran-Rcpp>0:devel/R-cran-Rcpp
BUILD_DEPENDS+=	${CRAN_DEPENDS}
RUN_DEPENDS=	${CRAN_DEPENDS}
TEST_DEPENDS=	R-cran-ggplot2>0:graphics/R-cran-ggplot2 \
		R-cran-jsonlite>0:converters/R-cran-jsonlite \
		R-cran-knitr>0:print/R-cran-knitr \
		R-cran-rmarkdown>0:textproc/R-cran-rmarkdown \
		R-cran-spelling>0:textproc/R-cran-spelling

USES=		compiler:c++17-lang cran:auto-plist,compiles magick:run

TESTING_UNSAFE=	Need 'gapminder'

.include <bsd.port.mk>
