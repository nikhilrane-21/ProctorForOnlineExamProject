# Generated by Django 3.0.14 on 2022-06-06 06:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('exam', '0007_auto_20220605_1431'),
    ]

    operations = [
        migrations.AddField(
            model_name='result',
            name='cheater',
            field=models.ImageField(blank=True, null=True, upload_to='C:/Users/TufA15/Desktop/Exam module/results/'),
        ),
    ]