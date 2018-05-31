from django.db import models

# Create your models here.

class Category(models.Model):
    name = models.CharField('카테고리 이름', max_length=40)


class Talk(models.Modle):
    category = models.ForeignKey(Category, verbose_name='카테고리')
    
