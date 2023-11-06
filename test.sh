coverage run -m --omit="tests/*" unittest discover -s ./tests -p 'test_*.py' -vv
coverage report -m