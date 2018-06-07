from django.db import models

# Create your models here.


class Category(models.Model):
    name = models.CharField('카테고리 이름', max_length=40)

    def __str__(self):
        return self.name


class Chat(models.Model):
    VERIFY_NOT = 0
    VERIFY_OK = 1
    VERIFY_DEL = 2

    CHOICES_VERIFY = (
        (VERIFY_NOT, '검토전'),
        (VERIFY_OK, '검토완료'),
        (VERIFY_DEL, '삭제필요'),
    )

    category = models.ForeignKey(Category, null=True, blank=True, on_delete=models.CASCADE)
    question = models.CharField(max_length=600, verbose_name='질문', help_text='한글 200자')
    answer = models.CharField(max_length=600, verbose_name='응답', help_text='한글 200자')
    verify = models.IntegerField(verbose_name='검토', help_text='0:검토전, 1:검토완료, 2:삭제필요', null=False, choices=CHOICES_VERIFY)
