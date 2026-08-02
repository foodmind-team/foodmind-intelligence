"""菜单 fixtures —— agent 测试的单一数据源（本地直连 graph / HTTP 联调共用）。

数据约定：
  - MENU_RECIPES       用户选定的完整菜单（7 道菜谱原文，含关键项补全前的原始缺口）
  - MOCK_INVENTORY     测试用 mock 库存快照（后端在真实场景中从自身数据库读取后
                       按本结构传入；这里保证每个资源充足，仅用于测试）
  - MOCK_KITCHEN_RESOURCES 测试用厨房设备快照（保证覆盖 decompose 所需资源类型）

注意：本文件与 scripts/test_recipes.py 中的内联数据互为副本；修改菜谱数据时请同步，
或后续统一迁移到本文件后删除 test_recipes.py 内的内联定义。
"""

from __future__ import annotations

# ============================================================================
# 菜单：7 道菜谱原文（用户选定 → 后端需按 GeneratePlanRequest.recipes 原样透传）
# ============================================================================

MENU_RECIPES = [
    {
        "recipe_id": "r1",
        "target_servings": 4,
        "text": """食材准备
主料：
蟹脚
辅料：
姜
蒜
洋葱
辣椒
辣椒面
火锅料
米酒
生抽
老抽
白胡椒粉、
酒糟
盐
葱段
烹饪步骤
1. 处理蟹脚：蟹脚洗净，用刷子刷净缝隙，刀背轻敲外壳方便入味。
2. 焯水去腥：冷水下锅加米酒焯水，捞出沥干。
3. 炒香底料：热油下姜蒜末、洋葱、辣椒炒香，加足量辣椒面和少许火锅料炒出红油。
4. 翻炒调味：放入蟹脚翻炒，加生抽、老抽、白胡椒粉、小半碗酒糟、少许盐炒匀。
5. 焖煮入味：加适量清水没过蟹脚，中火煮5-6分钟入味。
6. 出锅装盘：出锅前撒红辣椒、葱段即可装盘。
7. 享用：蟹脚吃完，汤汁拌米饭超下饭""",
    },
    {
        "recipe_id": "r2",
        "target_servings": 4,
        "text": """食材准备
主料：
鲜虾（适量，去头、去虾线、对半剪开）
辅料：
蒜末（大量，分两次用）
干辣椒
葱花
调料：
食用油
盐
鸡精
生抽
水淀粉
清水
烹饪步骤
处理虾：虾去头，拉出虾线，剪掉虾枪（虾头尖刺），从背部对半剪开，用清水洗净备用。
煎虾：锅中倒油烧热，下入处理好的虾，煎至外壳金黄焦脆，盛出备用。
炒香配料：锅中留底油（或重新倒油），油热后下入一半蒜末和干辣椒，大火炒出香味。
翻炒调味：倒入煎好的虾，快速翻炒均匀；加入适量盐、鸡精、生抽调味，继续翻炒至虾身裹上调料。
收汁入味：加入小半碗清水，煮至汤汁冒泡后，倒入少许水淀粉勾芡；最后加入剩余蒜末和葱花，大火快速翻炒均匀，即可出锅。""",
    },
    {
        "recipe_id": "r3",
        "target_servings": 4,
        "text": """食材准备
- 主料：
基围虾500克（选大个头）
- 辅料：
大蒜3-4瓣
小米辣（依吃辣程度放）
生姜2片
干辣椒（可选）
- 调料：
豆瓣酱1勺
生抽2勺
老抽少许
蚝油
黑胡椒粉
盐
味精/鸡精（可选）
食用油
烹饪步骤
1. 处理食材：
虾：剪去虾头，开背去除虾线，用厨房纸彻底吸干水分（防煎制溅油）
蒜：用重物拍扁，去皮后切成蒜末，备用。切成条或粒；
小米辣切圈；香菜切段；干辣椒可剪开（增香）。
2. 煎虾：热锅倒油，中小火加热至油微微热，放入控干水的虾，慢慢煎炒到虾壳变黄变焦，捞出备用。
3. 炒料：锅中补少许油，放姜、蒜、小米辣翻炒出香味，加1勺豆瓣酱炒出红油。
4. 调味：倒入煎好的虾，放干辣椒（可选），加2勺生抽、少许老抽（上色）、蚝油、黑胡椒粉、少许盐，可加味精提鲜，快速翻炒均匀。
5. 出锅""",
    },
    {
        "recipe_id": "r4",
        "target_servings": 4,
        "text": """食材准备
主料：
排骨
玉米
胡萝卜
辅料：
红枣
葱
姜
调料：
盐
烹饪步骤
焯水去腥：排骨冷水下锅，加入葱段、姜片焯水，撇去浮沫后捞出洗净。
放入食材：把焯好的排骨放入电饭煲，依次放入玉米段、胡萝卜块、山药块、红枣，加热水没过食材。
一键炖煮：盖好锅盖，选择煲汤模式，炖煮至程序结束。
调味出锅：出锅前撒入葱花，加适量盐调味，搅拌均匀即可。""",
    },
    {
        "recipe_id": "r5",
        "target_servings": 4,
        "text": """一、食材准备
- 主料
包菜1个
- 辅料：
大蒜20g
青花椒适量
干辣椒适量
- 调料：
盐1.5g
味精1g
白糖少许
生抽3g
香醋少许
二、制作步骤
1. 处理包菜：包菜从中间切开，去除硬梗，叶片逐片分开后手撕成大小均匀的片状，洗净后充分沥干水分备用。
2. 准备辅料：大蒜切蒜片，与青花椒、干辣椒一同放入碗中备用。
3. 爆香料头：锅烧热，倒入适量底油（可稍多防糊锅），油温烧至7成热，下入蒜片、干辣椒、青花椒，快速炸出香味。
4. 大火快炒：倒入沥干的包菜，大火翻炒30秒，炒至包菜断生（家庭灶可适当延长时间）。
5. 调味出锅：加入盐、味精、少许白糖提鲜，生抽从锅边淋入，加少许香醋，快速翻炒均匀，立即出锅装盘""",
    },
    {
        "recipe_id": "r6",
        "target_servings": 4,
        "text": """食材：
- 腊肠
- 菜花
- 蒜苗
- 蒜头
调料：
- 盐
鸡粉
蚝油
老抽
淀粉
生抽
猪油
米酒

步骤：
1. 腊肠水开后蒸15分钟，取出切片；菜花切小块，加两勺盐浸泡；蒜苗切段，蒜头切片。
2. 调个料汁：淀粉水+老抽+蚝油，混合备用。
3. 锅热不放油，下菜花反复煸炒5分钟，把水分煸干；盖盖焖1分钟，断生后盛出。
4. 锅中放猪油，下蒜片炒香，加入腊肠继续炒出香味。
5. 菜花回锅，加适量盐、鸡粉，沿锅边淋少许米酒。
6. 倒入调好的料汁，翻炒均匀，最后淋生抽提鲜出锅。""",
    },
    {
        "recipe_id": "r7",
        "target_servings": 4,
        "text": """食材准备
主料：
鸡翅中 15个
腌料：
盐
胡椒粉
鸡精
蚝油
老抽
料酒
蛋清 1个
淀粉 2大勺
辅料：
干辣椒（适量）
蒜末
火锅底料（半块）
白糖
白芝麻
点缀：
葱花（一把）
烹饪步骤
处理鸡翅
将15个鸡翅对半剪开（方便入味）。
加入盐、胡椒粉、鸡精、蚝油、老抽、料酒、1个蛋清、2大勺淀粉。抓拌均匀，腌制 20 分钟。
准备配料
干辣椒剪成丝，用清水浸泡一下（防止炒糊，更香）。
煎制鸡翅。热锅凉油，将腌好的鸡翅下锅。保持中火，煎至两面焦黄，散发香味。盛出鸡翅备用。
炒底料。再起锅，放入蒜末和半块火锅底料，炒化。倒入泡好的辣椒丝，炒出浓郁的香辣味。
混合翻炒。倒入煎好的鸡翅。加入 1勺白糖、五香粉、鸡精、白芝麻。最后放入一把葱花，大火快速翻炒均匀，即可出锅。""",
    },
]

# ============================================================================
# Mock 库存 —— 后端从数据库读取后按 InventoryLotSnapshot 结构传给 agent。
# 本文件保证每个资源充足，仅用于测试。
# ============================================================================

MOCK_INVENTORY = [
    # === 肉类/海鲜 ===
    {"lot_id": "l001", "item_id": "i001", "canonical_name": "蟹脚", "on_hand": "2000", "reserved": "0", "unit": "g"},
    {"lot_id": "l002", "item_id": "i002", "canonical_name": "鲜虾", "on_hand": "1000", "reserved": "0", "unit": "g"},
    {"lot_id": "l003", "item_id": "i003", "canonical_name": "基围虾", "on_hand": "800", "reserved": "0", "unit": "g"},
    {"lot_id": "l004", "item_id": "i004", "canonical_name": "排骨", "on_hand": "1000", "reserved": "0", "unit": "g"},
    {"lot_id": "l005", "item_id": "i005", "canonical_name": "鸡翅中", "on_hand": "15", "reserved": "0", "unit": "个"},
    {"lot_id": "l006", "item_id": "i006", "canonical_name": "腊肠", "on_hand": "4", "reserved": "0", "unit": "根"},
    {"lot_id": "l007", "item_id": "i007", "canonical_name": "鸡翅", "on_hand": "15", "reserved": "0", "unit": "个"},
    # === 蔬菜 ===
    {"lot_id": "l010", "item_id": "i010", "canonical_name": "包菜", "on_hand": "1", "reserved": "0", "unit": "个"},
    {"lot_id": "l011", "item_id": "i011", "canonical_name": "菜花", "on_hand": "1", "reserved": "0", "unit": "个"},
    {"lot_id": "l012", "item_id": "i012", "canonical_name": "玉米", "on_hand": "2", "reserved": "0", "unit": "根"},
    {"lot_id": "l013", "item_id": "i013", "canonical_name": "胡萝卜", "on_hand": "2", "reserved": "0", "unit": "根"},
    {"lot_id": "l014", "item_id": "i014", "canonical_name": "蒜苗", "on_hand": "3", "reserved": "0", "unit": "根"},
    {"lot_id": "l015", "item_id": "i015", "canonical_name": "洋葱", "on_hand": "2", "reserved": "0", "unit": "个"},
    {"lot_id": "l016", "item_id": "i016", "canonical_name": "山药", "on_hand": "500", "reserved": "0", "unit": "g"},
    # === 香料 ===
    {"lot_id": "l020", "item_id": "i020", "canonical_name": "姜", "on_hand": "200", "reserved": "0", "unit": "g"},
    {"lot_id": "l021", "item_id": "i021", "canonical_name": "生姜", "on_hand": "200", "reserved": "0", "unit": "g"},
    {"lot_id": "l022", "item_id": "i022", "canonical_name": "蒜", "on_hand": "300", "reserved": "0", "unit": "g"},
    {"lot_id": "l023", "item_id": "i023", "canonical_name": "大蒜", "on_hand": "100", "reserved": "0", "unit": "g"},
    {"lot_id": "l024", "item_id": "i024", "canonical_name": "蒜头", "on_hand": "100", "reserved": "0", "unit": "g"},
    {"lot_id": "l025", "item_id": "i025", "canonical_name": "蒜末", "on_hand": "200", "reserved": "0", "unit": "g"},
    {"lot_id": "l026", "item_id": "i026", "canonical_name": "葱", "on_hand": "100", "reserved": "0", "unit": "g"},
    {"lot_id": "l027", "item_id": "i027", "canonical_name": "葱段", "on_hand": "100", "reserved": "0", "unit": "g"},
    {"lot_id": "l028", "item_id": "i028", "canonical_name": "葱花", "on_hand": "100", "reserved": "0", "unit": "g"},
    {"lot_id": "l029", "item_id": "i029", "canonical_name": "干辣椒", "on_hand": "100", "reserved": "0", "unit": "g"},
    {"lot_id": "l030", "item_id": "i030", "canonical_name": "辣椒", "on_hand": "100", "reserved": "0", "unit": "g"},
    {"lot_id": "l031", "item_id": "i031", "canonical_name": "小米辣", "on_hand": "50", "reserved": "0", "unit": "g"},
    {"lot_id": "l032", "item_id": "i032", "canonical_name": "青花椒", "on_hand": "20", "reserved": "0", "unit": "g"},
    {"lot_id": "l033", "item_id": "i033", "canonical_name": "花椒", "on_hand": "20", "reserved": "0", "unit": "g"},
    {"lot_id": "l034", "item_id": "i034", "canonical_name": "红枣", "on_hand": "10", "reserved": "0", "unit": "个"},
    # === 调味品 ===
    {"lot_id": "l040", "item_id": "i040", "canonical_name": "盐", "on_hand": "500", "reserved": "0", "unit": "g"},
    {"lot_id": "l041", "item_id": "i041", "canonical_name": "生抽", "on_hand": "500", "reserved": "0", "unit": "ml"},
    {"lot_id": "l042", "item_id": "i042", "canonical_name": "老抽", "on_hand": "200", "reserved": "0", "unit": "ml"},
    {"lot_id": "l043", "item_id": "i043", "canonical_name": "蚝油", "on_hand": "300", "reserved": "0", "unit": "ml"},
    {"lot_id": "l044", "item_id": "i044", "canonical_name": "料酒", "on_hand": "300", "reserved": "0", "unit": "ml"},
    {"lot_id": "l045", "item_id": "i045", "canonical_name": "米酒", "on_hand": "300", "reserved": "0", "unit": "ml"},
    {"lot_id": "l046", "item_id": "i046", "canonical_name": "香醋", "on_hand": "200", "reserved": "0", "unit": "ml"},
    {"lot_id": "l047", "item_id": "i047", "canonical_name": "白糖", "on_hand": "300", "reserved": "0", "unit": "g"},
    {"lot_id": "l048", "item_id": "i048", "canonical_name": "豆瓣酱", "on_hand": "200", "reserved": "0", "unit": "g"},
    {"lot_id": "l049", "item_id": "i049", "canonical_name": "火锅底料", "on_hand": "200", "reserved": "0", "unit": "g"},
    {"lot_id": "l050", "item_id": "i050", "canonical_name": "火锅料", "on_hand": "200", "reserved": "0", "unit": "g"},
    {"lot_id": "l051", "item_id": "i051", "canonical_name": "辣椒面", "on_hand": "100", "reserved": "0", "unit": "g"},
    {"lot_id": "l052", "item_id": "i052", "canonical_name": "胡椒粉", "on_hand": "50", "reserved": "0", "unit": "g"},
    {"lot_id": "l053", "item_id": "i053", "canonical_name": "白胡椒粉", "on_hand": "50", "reserved": "0", "unit": "g"},
    {"lot_id": "l054", "item_id": "i054", "canonical_name": "黑胡椒粉", "on_hand": "50", "reserved": "0", "unit": "g"},
    {"lot_id": "l055", "item_id": "i055", "canonical_name": "鸡精", "on_hand": "100", "reserved": "0", "unit": "g"},
    {"lot_id": "l056", "item_id": "i056", "canonical_name": "味精", "on_hand": "100", "reserved": "0", "unit": "g"},
    {"lot_id": "l057", "item_id": "i057", "canonical_name": "鸡粉", "on_hand": "100", "reserved": "0", "unit": "g"},
    {"lot_id": "l058", "item_id": "i058", "canonical_name": "酒糟", "on_hand": "100", "reserved": "0", "unit": "g"},
    {"lot_id": "l059", "item_id": "i059", "canonical_name": "白芝麻", "on_hand": "50", "reserved": "0", "unit": "g"},
    {"lot_id": "l060", "item_id": "i060", "canonical_name": "五香粉", "on_hand": "30", "reserved": "0", "unit": "g"},
    # === 油脂/粉类 ===
    {"lot_id": "l070", "item_id": "i070", "canonical_name": "食用油", "on_hand": "1000", "reserved": "0", "unit": "ml"},
    {"lot_id": "l071", "item_id": "i071", "canonical_name": "猪油", "on_hand": "200", "reserved": "0", "unit": "g"},
    {"lot_id": "l072", "item_id": "i072", "canonical_name": "淀粉", "on_hand": "300", "reserved": "0", "unit": "g"},
    {"lot_id": "l073", "item_id": "i073", "canonical_name": "蛋清", "on_hand": "4", "reserved": "0", "unit": "个"},
    {"lot_id": "l074", "item_id": "i074", "canonical_name": "水淀粉", "on_hand": "200", "reserved": "0", "unit": "ml"},
    {"lot_id": "l075", "item_id": "i075", "canonical_name": "清水", "on_hand": "5000", "reserved": "0", "unit": "ml"},
    # === 覆盖节标题（规则解析器可能误提取为食材，测试用兜底） ===
    {"lot_id": "l080", "item_id": "i080", "canonical_name": "食材准备", "on_hand": "99", "reserved": "0", "unit": "个"},
    {"lot_id": "l081", "item_id": "i081", "canonical_name": "- 主料", "on_hand": "99", "reserved": "0", "unit": "个"},
    {"lot_id": "l082", "item_id": "i082", "canonical_name": "- 主料：", "on_hand": "99", "reserved": "0", "unit": "个"},
    {"lot_id": "l083", "item_id": "i083", "canonical_name": "- 辅料：", "on_hand": "99", "reserved": "0", "unit": "个"},
    {"lot_id": "l084", "item_id": "i084", "canonical_name": "- 调料：", "on_hand": "99", "reserved": "0", "unit": "个"},
    {"lot_id": "l085", "item_id": "i085", "canonical_name": "一、食材准备", "on_hand": "99", "reserved": "0", "unit": "个"},
    {"lot_id": "l086", "item_id": "i086", "canonical_name": "二、制作步骤", "on_hand": "99", "reserved": "0", "unit": "个"},
    {"lot_id": "l087", "item_id": "i087", "canonical_name": "- 盐", "on_hand": "99", "reserved": "0", "unit": "个"},
    {"lot_id": "l088", "item_id": "i088", "canonical_name": "调料：", "on_hand": "99", "reserved": "0", "unit": "个"},
    {"lot_id": "l089", "item_id": "i089", "canonical_name": "腌料：", "on_hand": "99", "reserved": "0", "unit": "个"},
    {"lot_id": "l090", "item_id": "i090", "canonical_name": "点缀：", "on_hand": "99", "reserved": "0", "unit": "个"},
]

# ============================================================================
# Mock 厨房设备 —— 覆盖 decompose 阶段产生的 ResourceNeed 类型
# ============================================================================

MOCK_KITCHEN_RESOURCES = [
    {"resource_id": "r1", "resource_type": "stove", "capacity": "2", "capacity_unit": "burners", "capabilities": ["gas"], "available": True},
    {"resource_id": "r2", "resource_type": "stove", "capacity": "2", "capacity_unit": "burners", "capabilities": ["gas"], "available": True},
    {"resource_id": "r3", "resource_type": "oven", "capacity": "1", "capacity_unit": "racks", "capabilities": ["convection"], "available": True},
    {"resource_id": "r4", "resource_type": "sink", "capacity": "2", "capacity_unit": "basins", "capabilities": [], "available": True},
    {"resource_id": "r5", "resource_type": "rice_cooker", "capacity": "1", "capacity_unit": "pots", "capabilities": ["stew"], "available": True},
    {"resource_id": "r6", "resource_type": "knife", "capacity": "3", "capacity_unit": "pieces", "capabilities": ["cutting", "chopping"], "available": True},
    {"resource_id": "r7", "resource_type": "pot", "capacity": "4", "capacity_unit": "pieces", "capabilities": ["boiling", "stewing", "stir_frying"], "available": True},
    {"resource_id": "r8", "resource_type": "cutting_board", "capacity": "2", "capacity_unit": "pieces", "capabilities": ["cutting"], "available": True},
    {"resource_id": "r9", "resource_type": "wok", "capacity": "2", "capacity_unit": "pieces", "capabilities": ["stir_frying", "deep_frying"], "available": True},
    {"resource_id": "r10", "resource_type": "spatula", "capacity": "3", "capacity_unit": "pieces", "capabilities": ["stirring", "flipping"], "available": True},
    {"resource_id": "r11", "resource_type": "mixing_bowl", "capacity": "3", "capacity_unit": "pieces", "capabilities": ["mixing", "marinating"], "available": True},
]
