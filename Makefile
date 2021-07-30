RUN_DATE=$(shell python -c 'from datetime import datetime; print(datetime.today().strftime("%Y-%m-%d"))')

default:
	@echo $(DATE)

publish: convert sync

convert:
	jupyter nbconvert sydney-outbreaks.ipynb --to html --stdout > archive/$(RUN_DATE)/sydney-outbreaks.html

sync:
	aws s3 sync archive/$(RUN_DATE)/ s3://jonseymour/delta-waratah/archive/$(RUN_DATE)/ --profile jonseymour --acl public-read
	aws s3 sync archive/$(RUN_DATE)/ s3://jonseymour/delta-waratah/archive/latest/ --profile jonseymour --acl public-read

sync-archive:
	aws s3 sync archive/ s3://jonseymour/delta-waratah/archive/ --profile jonseymour --acl public-read
	