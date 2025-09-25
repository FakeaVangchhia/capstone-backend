from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("chat", "0002_appuser_authtoken"),
    ]

    operations = [
        migrations.AddField(
            model_name="appuser",
            name="is_admin",
            field=models.BooleanField(default=False),
        ),
    ]


