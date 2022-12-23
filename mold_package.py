import numpy as np
import os, sys
from datetime import date

today = date.today()

template_name = "SimplePythonPackageTemplate"
template_author = 'The Author'
template_email  = 'the.author@website.com'
template_date   = '25 November 2022'
template_user_name = 'the-author'

package_name = input('Enter your package name: ')
package_name = 'SimplePythonPackageTemplate' if len(package_name)==0 else package_name
package_description_line = input('Enter a one line description of your package: ')
package_description_line = 'This is a simple python package template that can be forked and used.' \
                                if len(package_description_line)==0 else package_description_line
user_name = input('Enter the user name: ')
user_name = 'the-author' if len(user_name)==0 else user_name
author_name = input('Enter the author name: ')
author_name = 'The Author' if len(author_name)==0 else author_name
author_email = input('Enter your author email: ')
author_email = 'the.author@website.com' if len(author_email)==0 else author_email
date_created = input('Enter the date when the package was created: ')
date_created = str(today) if len(date_created)==0 else date_created

print(''.join(['-' for i in range(100)]))
print(package_name)
print(package_description_line)
print(author_name)
print(author_email)
print(date_created)
print(''.join(['-' for i in range(100)]))

filename = 'README.md'
print('Molding {} ...'.format(filename))
with open(filename, 'r') as f:
    file = f.readlines()
for ii, line in enumerate(file): file[ii] = line.replace(template_name, package_name)
for ii, line in enumerate(file): file[ii] = line.replace(template_author, author_name)
for ii, line in enumerate(file): file[ii] = line.replace(template_email, author_email)
is_new_package = False
for ii, line in enumerate(file): 
    if 'If you are using this template, please mention the following in your package:' in line:
        if package_name != 'SimplePythonPackageTemplate': file[ii] = ''
    if 'This package uses the template provided at https://github.com/sambit-giri/' in line:
        if package_name != 'SimplePythonPackageTemplate': file[ii] = line.replace(package_name, template_name)
file[2] = file[2].replace(file[2].split('. More documentation')[0], package_description_line)
with open(filename, 'w') as f:
    f.writelines(file)
print('...done')

filename = 'setup.py'
print('Molding {} ...'.format(filename))
with open(filename, 'r') as f:
    file = f.readlines()
for ii, line in enumerate(file): file[ii] = line.replace(template_name, package_name)
for ii, line in enumerate(file): file[ii] = line.replace(template_author, author_name)
for ii, line in enumerate(file): file[ii] = line.replace(template_email, author_email)
for ii, line in enumerate(file): file[ii] = line.replace('Created on {}'.format(template_date), 'Created on {}'.format(date_created))
with open(filename, 'w') as f:
    f.writelines(file)
print('...done')

filename = 'CONTRIBUTING.rst'
print('Molding {} ...'.format(filename))
with open(filename, 'r') as f:
    file = f.readlines()
for ii, line in enumerate(file):
    file[ii] = line.replace(template_name, package_name)
    # file[ii] = line.replace(template_author, author_name)
    # file[ii] = line.replace(template_email, author_email)
    # file[ii] = line.replace('Created on {}'.format(template_date), 'Created on {}'.format(date_created))
with open(filename, 'w') as f:
    f.writelines(file)
print('...done')

filename = 'tests/test_calculator.py'
print('Molding {} ...'.format(filename))
with open(filename, 'r') as f:
    file = f.readlines()
for ii, line in enumerate(file): file[ii] = line.replace(template_name, package_name)
# for ii, line in enumerate(file): file[ii] = line.replace(template_author, author_name)
# for ii, line in enumerate(file): file[ii] = line.replace(template_email, author_email)
# for ii, line in enumerate(file): file[ii] = line.replace('Created on {}'.format(template_date), 'Created on {}'.format(date_created))
with open(filename, 'w') as f:
    f.writelines(file)
print('...done')

filename = 'examples/age_of_universe.py'
print('Molding {} ...'.format(filename))
with open(filename, 'r') as f:
    file = f.readlines()
for ii, line in enumerate(file): file[ii] = line.replace(template_name, package_name)
# for ii, line in enumerate(file): file[ii] = line.replace(template_author, author_name)
# for ii, line in enumerate(file): file[ii] = line.replace(template_email, author_email)
# for ii, line in enumerate(file): file[ii] = line.replace('Created on {}'.format(template_date), 'Created on {}'.format(date_created))
with open(filename, 'w') as f:
    f.writelines(file)
print('...done')

src_path_old = 'src/{}'.format(template_name)
src_path_new = 'src/{}'.format(package_name)
print('Preparing the folder that would contain the code...')
try:
    os.rename(src_path_old, src_path_new)
    print('{} renamed to {}'.format(src_path_old,src_path_new))
except:
    if not os.path.exists(src_path_new):
        os.mkdir(src_path_new)
        print('{} folder created'.format(src_path_new))
    else:
        print('{} folder already exists'.format(src_path_new))
print('...done')


print('Modifying the docs folder.')

filename = 'docs/authors.rst'
print('Molding {} ...'.format(filename))
with open(filename, 'r') as f:
    file = f.readlines()
for ii, line in enumerate(file): file[ii] = line.replace(template_name, package_name)
for ii, line in enumerate(file): file[ii] = line.replace(template_user_name, user_name)
with open(filename, 'w') as f:
    f.writelines(file)
print('...done')

filename = 'docs/installation.rst'
print('Molding {} ...'.format(filename))
with open(filename, 'r') as f:
    file = f.readlines()
for ii, line in enumerate(file): file[ii] = line.replace(template_name, package_name)
for ii, line in enumerate(file): file[ii] = line.replace(template_user_name, user_name)
with open(filename, 'w') as f:
    f.writelines(file)
print('...done')

filename = 'docs/readme.rst'
print('Molding {} ...'.format(filename))
with open(filename, 'r') as f:
    file = f.readlines()
for ii, line in enumerate(file): file[ii] = line.replace(template_name, package_name)
for ii, line in enumerate(file): file[ii] = line.replace(template_user_name, user_name)
with open(filename, 'w') as f:
    f.writelines(file)
print('...done')

filename = 'docs/tutorials.rst'
print('Molding {} ...'.format(filename))
with open(filename, 'r') as f:
    file = f.readlines()
for ii, line in enumerate(file): file[ii] = line.replace(template_name, package_name)
for ii, line in enumerate(file): file[ii] = line.replace(template_user_name, user_name)
with open(filename, 'w') as f:
    f.writelines(file)
print('...done')

print('The templated is prepared for your package.')
