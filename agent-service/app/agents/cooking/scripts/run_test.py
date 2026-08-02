"""Direct workflow test — no triple-quoted strings."""

import asyncio
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["COOKING_PLAN_INTERNAL_SERVICE_TOKEN"] = "test-token"

from decimal import Decimal

from cooking_plan_agent.domain.models import GeneratePlanRequest, InventoryLotSnapshot, KitchenResourceSnapshot
from cooking_plan_agent.safety.engine import SafetyEngine
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.graph import build_cooking_plan_graph


def R(rid, text):  # noqa: N802 — compact recipe factory for the debug harness
    return {"recipe_id": rid, "target_servings": 4, "text": text}


RECIPES = [
    R(
        "r1",
        "食材准备\n主料：\n蟹脚\n辅料：\n姜\n蒜\n洋葱\n辣椒\n辣椒面\n火锅料\n调料：\n米酒\n生抽\n老抽\n白胡椒粉\n酒糟\n盐\n葱\n烹饪步骤\n1. 处理蟹脚：蟹脚洗净，用刷子刷净缝隙，刀背轻敲外壳方便入味。\n2. 焯水去腥：冷水下锅加米酒焯水，捞出沥干。\n3. 炒香底料：热油下姜蒜末、洋葱、辣椒炒香，加足量辣椒面和少许火锅料炒出红油。\n4. 翻炒调味：放入蟹脚翻炒，加生抽、老抽、白胡椒粉、小半碗酒糟、少许盐炒匀。\n5. 焖煮入味：加适量清水没过蟹脚，中火炖5-6分钟入味。\n6. 出锅装盘：出锅前撒红辣椒、葱段即可装盘。",
    ),
    R(
        "r2",
        "食材准备\n主料：\n鲜虾\n辅料：\n蒜\n干辣椒\n葱\n调料：\n食用油\n盐\n鸡精\n生抽\n水淀粉\n烹饪步骤\n1. 处理虾：虾去头，拉出虾线，剪掉虾枪，从背部对半剪开，用清水洗净备用。\n2. 煎虾：锅中倒油烧热，下入处理好的虾，煎至外壳金黄焦脆，盛出备用。\n3. 炒香配料：锅中留底油，油热后下入一半蒜末和干辣椒，大火炒出香味。\n4. 翻炒调味：倒入煎好的虾，快速翻炒均匀，加入盐、鸡精、生抽调味。\n5. 收汁入味：加入小半碗清水，煮至汤汁冒泡后，倒入少许水淀粉勾芡，加入剩余蒜末和葱花翻炒出锅。",
    ),
    R(
        "r3",
        "食材准备\n主料：\n基围虾\n辅料：\n大蒜\n小米辣\n生姜\n干辣椒\n调料：\n豆瓣酱\n生抽\n老抽\n蚝油\n黑胡椒粉\n盐\n味精\n食用油\n烹饪步骤\n1. 处理食材：虾剪去虾头，开背去除虾线，用厨房纸彻底吸干水分。蒜拍扁去皮切成蒜末。小米辣切圈，干辣椒剪开。\n2. 煎虾：热锅倒油，中小火加热至油微微热，放入控干水的虾，慢慢煎炒到虾壳变黄变焦，捞出备用。\n3. 炒料：锅中补少许油，放姜、蒜、小米辣翻炒出香味，加1勺豆瓣酱炒出红油。\n4. 调味：倒入煎好的虾，放干辣椒，加生抽、老抽、蚝油、黑胡椒粉、盐、味精，快速翻炒均匀。\n5. 出锅装盘。",
    ),
    R(
        "r4",
        "食材准备\n主料：\n排骨\n玉米\n胡萝卜\n辅料：\n红枣\n葱\n姜\n调料：\n盐\n烹饪步骤\n1. 焯水去腥：排骨冷水下锅，加入葱段、姜片焯水，撇去浮沫后捞出洗净。\n2. 放入食材：把焯好的排骨放入电饭煲，依次放入玉米段、胡萝卜块、红枣，加热水没过食材。\n3. 一键炖煮：盖好锅盖，选择煲汤模式，炖煮60分钟。\n4. 调味出锅：出锅前撒入葱花，加适量盐调味，搅拌均匀即可。",
    ),
    R(
        "r5",
        "食材准备\n主料：\n包菜\n辅料：\n大蒜\n青花椒\n干辣椒\n调料：\n盐\n味精\n白糖\n生抽\n香醋\n烹饪步骤\n1. 处理包菜：包菜从中间切开，去除硬梗，叶片逐片分开后手撕成大小均匀的片状，洗净后充分沥干水分备用。\n2. 准备辅料：大蒜切蒜片，与青花椒、干辣椒一同放入碗中备用。\n3. 爆香料头：锅烧热，倒入适量底油，油温烧至7成热，下入蒜片、干辣椒、青花椒，快速炸出香味。\n4. 大火快炒：倒入沥干的包菜，大火翻炒30秒，炒至包菜断生。\n5. 调味出锅：加入盐、味精、白糖提鲜，生抽从锅边淋入，加香醋，快速翻炒均匀，立即出锅装盘。",
    ),
    R(
        "r6",
        "食材准备\n主料：\n腊肠\n菜花\n辅料：\n蒜苗\n蒜头\n调料：\n盐\n鸡粉\n蚝油\n老抽\n淀粉\n生抽\n猪油\n米酒\n烹饪步骤\n1. 腊肠水开后蒸15分钟，取出切片；菜花切小块，加两勺盐浸泡；蒜苗切段，蒜头切片。\n2. 调个料汁：淀粉水加老抽加蚝油，混合备用。\n3. 锅热不放油，下菜花反复煸炒5分钟，把水分煸干，盖盖焖1分钟，断生后盛出。\n4. 锅中放猪油，下蒜片炒香，加入腊肠继续炒出香味。\n5. 菜花回锅，加盐、鸡粉，沿锅边淋米酒。\n6. 倒入调好的料汁，翻炒均匀，最后淋生抽提鲜出锅。",
    ),
    R(
        "r7",
        "食材准备\n主料：\n鸡翅中\n辅料：\n干辣椒\n蒜\n火锅底料\n白糖\n白芝麻\n葱\n调料：\n盐\n胡椒粉\n鸡精\n蚝油\n老抽\n料酒\n蛋清\n淀粉\n烹饪步骤\n1. 处理鸡翅：将15个鸡翅对半剪开。加入盐、胡椒粉、鸡精、蚝油、老抽、料酒、蛋清、淀粉，抓拌均匀，腌制20分钟。\n2. 准备配料：干辣椒剪成丝，用清水浸泡一下防止炒糊。\n3. 煎制鸡翅：热锅凉油，将腌好的鸡翅下锅，保持中火，煎至两面焦黄，散发香味，盛出鸡翅备用。\n4. 炒底料：再起锅，放入蒜末和半块火锅底料，炒化，倒入泡好的辣椒丝，炒出浓郁的香辣味。\n5. 混合翻炒：倒入煎好的鸡翅，加入白糖、五香粉、鸡精、白芝麻，最后放入葱花，大火快速翻炒均匀，即可出锅。",
    ),
]


def L(name, unit, qty="500"):  # noqa: N802 — compact inventory-lot factory for the debug harness
    return InventoryLotSnapshot(
        lot_id=f"l_{name}",
        item_id=f"i_{name}",
        canonical_name=name,
        on_hand=Decimal(qty),
        reserved=Decimal(0),
        unit=unit,
    )


INVENTORY = (
    L("蟹脚", "g", "2000"),
    L("鲜虾", "g", "1000"),
    L("基围虾", "g", "800"),
    L("排骨", "g", "1000"),
    L("鸡翅中", "个", "30"),
    L("腊肠", "根", "4"),
    L("包菜", "个", "2"),
    L("菜花", "个", "2"),
    L("玉米", "根", "3"),
    L("胡萝卜", "根", "3"),
    L("蒜苗", "根", "5"),
    L("洋葱", "个", "2"),
    L("姜", "g"),
    L("生姜", "g"),
    L("蒜", "g", "500"),
    L("大蒜", "g"),
    L("蒜头", "g"),
    L("蒜末", "g"),
    L("葱", "g", "300"),
    L("葱段", "g"),
    L("葱花", "g"),
    L("干辣椒", "g", "200"),
    L("辣椒", "g"),
    L("小米辣", "g"),
    L("青花椒", "g"),
    L("红枣", "个", "20"),
    L("盐", "g", "1000"),
    L("生抽", "ml"),
    L("老抽", "ml"),
    L("蚝油", "ml"),
    L("料酒", "ml"),
    L("米酒", "ml"),
    L("香醋", "ml"),
    L("白糖", "g"),
    L("豆瓣酱", "g"),
    L("火锅底料", "g", "300"),
    L("火锅料", "g", "300"),
    L("辣椒面", "g"),
    L("胡椒粉", "g"),
    L("白胡椒粉", "g"),
    L("黑胡椒粉", "g"),
    L("鸡精", "g"),
    L("味精", "g"),
    L("鸡粉", "g"),
    L("酒糟", "g"),
    L("白芝麻", "g"),
    L("五香粉", "g"),
    L("食用油", "ml", "2000"),
    L("猪油", "g"),
    L("淀粉", "g"),
    L("蛋清", "个", "10"),
    L("水淀粉", "ml"),
)

RESOURCES = (
    KitchenResourceSnapshot(
        resource_id="r1",
        resource_type="stove",
        capacity=Decimal(2),
        capacity_unit="burners",
        capabilities=("gas",),
        available=True,
    ),
    KitchenResourceSnapshot(
        resource_id="r2",
        resource_type="stove",
        capacity=Decimal(2),
        capacity_unit="burners",
        capabilities=("gas",),
        available=True,
    ),
    KitchenResourceSnapshot(
        resource_id="r3",
        resource_type="oven",
        capacity=Decimal(1),
        capacity_unit="racks",
        capabilities=("convection",),
        available=True,
    ),
    KitchenResourceSnapshot(
        resource_id="r4",
        resource_type="sink",
        capacity=Decimal(2),
        capacity_unit="basins",
        capabilities=(),
        available=True,
    ),
    KitchenResourceSnapshot(
        resource_id="r5",
        resource_type="rice_cooker",
        capacity=Decimal(1),
        capacity_unit="pots",
        capabilities=("stew",),
        available=True,
    ),
    KitchenResourceSnapshot(
        resource_id="r6",
        resource_type="knife",
        capacity=Decimal(3),
        capacity_unit="pieces",
        capabilities=("cutting", "chopping"),
        available=True,
    ),
    KitchenResourceSnapshot(
        resource_id="r7",
        resource_type="pot",
        capacity=Decimal(4),
        capacity_unit="pieces",
        capabilities=("boiling", "stewing", "stir_frying"),
        available=True,
    ),
    KitchenResourceSnapshot(
        resource_id="r8",
        resource_type="cutting_board",
        capacity=Decimal(2),
        capacity_unit="pieces",
        capabilities=("cutting",),
        available=True,
    ),
    KitchenResourceSnapshot(
        resource_id="r9",
        resource_type="wok",
        capacity=Decimal(2),
        capacity_unit="pieces",
        capabilities=("stir_frying", "deep_frying"),
        available=True,
    ),
    KitchenResourceSnapshot(
        resource_id="r10",
        resource_type="spatula",
        capacity=Decimal(3),
        capacity_unit="pieces",
        capabilities=("stirring", "flipping"),
        available=True,
    ),
    KitchenResourceSnapshot(
        resource_id="r11",
        resource_type="mixing_bowl",
        capacity=Decimal(3),
        capacity_unit="pieces",
        capabilities=("mixing", "marinating"),
        available=True,
    ),
)


async def main():
    request = GeneratePlanRequest(
        request_id="t",
        user_id="u",
        recipes=RECIPES,
        inventory_lots=INVENTORY,
        kitchen_resources=RESOURCES,
    )
    graph = build_cooking_plan_graph()
    ctx = WorkflowContext(recipe_extractor=None, recipe_researcher=None, safety_engine=SafetyEngine())
    try:
        result = await graph.ainvoke({"request": request}, context=ctx)
    except Exception:  # noqa: BLE001 — debug harness: always surface the full traceback
        traceback.print_exc()
        return
    # Debug intermediate states
    for k in ["verification_report", "schedule_result", "task_graph", "recipe_tasks"]:
        v = result.get(k)
        if v is not None:
            print(f"[{k}]: {type(v).__name__}", end="")
            if hasattr(v, "passed"):
                print(f" passed={v.passed} issues={len(v.issues) if hasattr(v, 'issues') else '?'}")
                for iss in getattr(v, "issues", ())[:5]:
                    print(f"  - {iss.code}: {iss.message}")
            elif hasattr(v, "status"):
                print(f" status={v.status}")
            elif hasattr(v, "__len__"):
                print(f" len={len(v)}")
            else:
                print()
    err = result.get("error")
    if err:
        print(f"ERROR [{err.node_name}]: {err.error_code} - {err.message}")
    resp = result.get("response")
    if not resp:
        print("NO RESPONSE:", sorted(result.keys()))
        return
    print(f"\n===== STATUS: {resp.status} =====")
    if resp.status == "READY":
        print(f"总耗时: {resp.makespan_minutes} 分钟")
        print(f"\n时间线 ({len(resp.timeline)} 任务):")
        for t in resp.timeline:
            print(f"  [{t['start_minute']:>4}-{t['end_minute']:>4}] {t['task_id'][:55]}: {t['instruction'][:60]}")
        print("\n各菜完成时间:")
        for d in sorted(resp.dish_completions, key=lambda x: x.get("completion_minute", 0)):
            rid = d.get("recipe_id", d.get("dish_id", "?"))
            print(f"  {rid}: {d.get('completion_minute', '?')} 分钟")
    elif resp.status == "NEEDS_CONFIRMATION":
        print(f"假设: {len(resp.assumptions)}, 修复选项: {len(resp.repair_options)}")
        for r in resp.repair_options[:5]:
            print(f"  - {r['description'][:100]}")
    elif resp.status == "FAILED":
        print(f"{resp.error_code}: {resp.message}")
    else:
        print(f"Unknown: {resp}")


asyncio.run(main())
