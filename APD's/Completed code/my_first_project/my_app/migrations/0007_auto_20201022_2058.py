# Generated by Django 3.1.1 on 2020-10-22 15:28

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('my_app', '0006_post'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='post',
            new_name='post_table',
        ),
    ]