# Generated by Django 4.1.2 on 2022-11-01 04:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('MyAPI', '0014_alter_user_image'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='image',
            field=models.ImageField(blank=True, default='profile_image/profile.png', upload_to='profile_image'),
        ),
    ]
